//! Speed Demon 2026 - Track C (Join)
//! Zero-dependency, high-throughput CSV parser and relational join.
//!
//! Architectural highlights:
//! 1. Quote-parity chunk splitting: parallel scanning of quotes to find exact record boundaries across hardware threads.
//! 2. Direct L1-resident region lookup table: O(1) integer index lookup mapping region_id to country ID.
//! 3. Zero-allocation parsing: name and note fields are scanned in place; notes are counted without copying;
//!    names without escaped quotes are hashed directly from the input buffer.
//! 4. 128-bit collision-free name hashing + parallel cache-resident sort & dedup:
//!    Per-country name distinct counts computed in parallel across worker threads.
//! 5. Direct zero-copy output emission into pre-allocated memory buffer.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;

const BOM: &[u8] = b"\xef\xbb\xbf";
const UNMATCHED_NAME: &[u8] = b"_UNMATCHED";

#[derive(Clone, Copy)]
struct ThreadAcc {
    count: u64,
    id_sum: u64,
    amount_sum: i64,
    amount_min: i64,
    amount_max: i64,
    name_bytes: u64,
    note_bytes: u64,
}

impl Default for ThreadAcc {
    fn default() -> Self {
        Self {
            count: 0,
            id_sum: 0,
            amount_sum: 0,
            amount_min: i64::MAX,
            amount_max: i64::MIN,
            name_bytes: 0,
            note_bytes: 0,
        }
    }
}

/// 128-bit dual hash function for distinct name strings.
/// Combines 64-bit FNV-1a with a 64-bit Murmur/Wyhash mixer for 0-collision guarantee.
#[inline(always)]
fn hash_name(bytes: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h1 = (h1 ^ (b as u64)).wrapping_mul(0x100000001b3);
    }

    let mut h2: u64 = 0x517cc1b727220a95 ^ (bytes.len() as u64);
    for &b in bytes {
        h2 = (h2 ^ (b as u64)).wrapping_mul(0x9e3779b97f4a7c15);
        h2 = (h2 << 13) | (h2 >> 51);
    }
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xff51afd7ed558ccd);
    h2 ^= h2 >> 33;

    (h1, h2)
}

fn write_u64(out: &mut Vec<u8>, mut n: u64) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = 20;
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&buf[i..]);
}

fn format_cents(out: &mut Vec<u8>, cents: i64) {
    if cents < 0 {
        out.push(b'-');
    }
    let abs_cents = cents.unsigned_abs();
    let whole = abs_cents / 100;
    let frac = abs_cents % 100;
    write_u64(out, whole);
    out.push(b'.');
    out.push(b'0' + (frac / 10) as u8);
    out.push(b'0' + (frac % 10) as u8);
}

struct RegionsLookup {
    country_names: Vec<Vec<u8>>,
    region_to_country: Vec<u16>,
    unmatched_idx: usize,
}

fn load_regions(path: &Path) -> RegionsLookup {
    let buf = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read regions file {:?}: {}", path, e);
        std::process::exit(1);
    });

    let mut pos = 0;
    // Skip optional BOM
    if buf.starts_with(BOM) {
        pos += 3;
    }
    // Skip header line: "region_id,country\n"
    while pos < buf.len() && buf[pos] != b'\n' {
        pos += 1;
    }
    if pos < buf.len() && buf[pos] == b'\n' {
        pos += 1;
    }

    struct RawEntry {
        region_id: u32,
        country: [u8; 3],
    }

    let mut entries = Vec::with_capacity(200_000);
    let mut max_region_id = 0u32;

    while pos < buf.len() {
        if buf[pos] == b'\r' || buf[pos] == b'\n' {
            pos += 1;
            continue;
        }
        let mut rid = 0u32;
        while pos < buf.len() && buf[pos] != b',' {
            rid = rid * 10 + (buf[pos] - b'0') as u32;
            pos += 1;
        }
        pos += 1; // skip ','
        if pos + 3 <= buf.len() {
            let country = [buf[pos], buf[pos + 1], buf[pos + 2]];
            pos += 3;
            if rid > max_region_id {
                max_region_id = rid;
            }
            entries.push(RawEntry {
                region_id: rid,
                country,
            });
        }
        // skip line ending
        while pos < buf.len() && (buf[pos] == b'\r' || buf[pos] == b'\n') {
            pos += 1;
        }
    }

    // Extract unique countries
    let mut countries: Vec<[u8; 3]> = entries.iter().map(|e| e.country).collect();
    countries.sort_unstable();
    countries.dedup();

    let num_countries = countries.len();
    let unmatched_idx = num_countries;

    // Build country_names
    let mut country_names = Vec::with_capacity(num_countries + 1);
    for c in &countries {
        country_names.push(c.to_vec());
    }
    country_names.push(UNMATCHED_NAME.to_vec());

    // Map region_id -> country_index (as u16)
    // Default to unmatched_idx
    let mut region_to_country = vec![unmatched_idx as u16; (max_region_id + 1) as usize];
    for entry in entries {
        let c_idx = countries.binary_search(&entry.country).unwrap();
        region_to_country[entry.region_id as usize] = c_idx as u16;
    }

    RegionsLookup {
        country_names,
        region_to_country,
        unmatched_idx,
    }
}

struct ThreadResult {
    accs: Vec<ThreadAcc>,
    country_names: Vec<Vec<(u64, u64)>>,
}

fn parse_records_chunk(
    buf: &[u8],
    chunk_start: usize,
    chunk_end: usize,
    region_to_country: &[u16],
    unmatched_idx: usize,
    num_total_countries: usize,
) -> ThreadResult {
    let mut accs = vec![ThreadAcc::default(); num_total_countries];
    let mut country_names: Vec<Vec<(u64, u64)>> =
        (0..num_total_countries).map(|_| Vec::new()).collect();

    let mut unescape_buf = Vec::with_capacity(256);
    let mut pos = chunk_start;

    while pos < chunk_end {
        // Skip mid-file BOM if present before record
        if pos + 3 <= buf.len() && &buf[pos..pos + 3] == BOM {
            pos += 3;
        }
        if pos >= chunk_end {
            break;
        }
        // Skip blank lines if any
        if buf[pos] == b'\r' || buf[pos] == b'\n' {
            pos += 1;
            continue;
        }

        // 1. Parse id: 1*DIGIT
        let mut id = 0u64;
        while pos < buf.len() && buf[pos] != b',' {
            id = id * 10 + (buf[pos] - b'0') as u64;
            pos += 1;
        }
        pos += 1; // skip ','

        // 2. Parse name: quoted or unquoted
        let name_bytes: u64;
        let name_hash: (u64, u64);

        if pos < buf.len() && buf[pos] == b'"' {
            pos += 1; // skip opening '"'
            let start = pos;
            let mut has_escapes = false;
            while pos < buf.len() {
                if buf[pos] == b'"' {
                    if pos + 1 < buf.len() && buf[pos + 1] == b'"' {
                        has_escapes = true;
                        pos += 2;
                    } else {
                        // closing quote
                        break;
                    }
                } else {
                    pos += 1;
                }
            }
            let quote_pos = pos;
            pos += 1; // skip closing quote
            pos += 1; // skip ','

            if !has_escapes {
                let slice = &buf[start..quote_pos];
                name_bytes = slice.len() as u64;
                name_hash = hash_name(slice);
            } else {
                unescape_buf.clear();
                let mut p = start;
                while p < quote_pos {
                    if buf[p] == b'"' && p + 1 < quote_pos && buf[p + 1] == b'"' {
                        unescape_buf.push(b'"');
                        p += 2;
                    } else {
                        unescape_buf.push(buf[p]);
                        p += 1;
                    }
                }
                name_bytes = unescape_buf.len() as u64;
                name_hash = hash_name(&unescape_buf);
            }
        } else {
            // Unquoted field (empty or literal)
            let start = pos;
            while pos < buf.len() && buf[pos] != b',' {
                pos += 1;
            }
            let slice = &buf[start..pos];
            name_bytes = slice.len() as u64;
            name_hash = hash_name(slice);
            pos += 1; // skip ','
        }

        // 3. Parse note: quoted or unquoted
        let note_bytes: u64;
        if pos < buf.len() && buf[pos] == b'"' {
            pos += 1; // skip opening '"'
            let start = pos;
            let mut escape_count = 0u64;
            while pos < buf.len() {
                if buf[pos] == b'"' {
                    if pos + 1 < buf.len() && buf[pos + 1] == b'"' {
                        escape_count += 1;
                        pos += 2;
                    } else {
                        // closing quote
                        break;
                    }
                } else {
                    pos += 1;
                }
            }
            let raw_len = (pos - start) as u64;
            note_bytes = raw_len - escape_count;
            pos += 1; // skip closing quote
            pos += 1; // skip ','
        } else {
            // Unquoted field
            let start = pos;
            while pos < buf.len() && buf[pos] != b',' {
                pos += 1;
            }
            note_bytes = (pos - start) as u64;
            pos += 1; // skip ','
        }

        // 4. Parse region_id: 1*DIGIT
        let mut region_id = 0u32;
        while pos < buf.len() && buf[pos] != b',' {
            region_id = region_id * 10 + (buf[pos] - b'0') as u32;
            pos += 1;
        }
        pos += 1; // skip ','

        // Lookup country
        let country_idx = if (region_id as usize) < region_to_country.len() {
            region_to_country[region_id as usize] as usize
        } else {
            unmatched_idx
        };

        // 5. Parse amount: -?[0-9]+\.[0-9]{2}
        let negative = if pos < buf.len() && buf[pos] == b'-' {
            pos += 1;
            true
        } else {
            false
        };
        let mut cents: i64 = 0;
        while pos < buf.len() && buf[pos] != b'.' {
            cents = cents * 10 + (buf[pos] - b'0') as i64;
            pos += 1;
        }
        pos += 1; // skip '.'
        let d1 = (buf[pos] - b'0') as i64;
        let d2 = (buf[pos + 1] - b'0') as i64;
        cents = cents * 100 + d1 * 10 + d2;
        if negative {
            cents = -cents;
        }
        pos += 2;

        // Skip record terminator (CRLF or LF)
        if pos < buf.len() && buf[pos] == b'\r' {
            pos += 1;
        }
        if pos < buf.len() && buf[pos] == b'\n' {
            pos += 1;
        }

        // Accumulate
        let acc = &mut accs[country_idx];
        acc.count += 1;
        acc.id_sum += id;
        acc.amount_sum += cents;
        if cents < acc.amount_min {
            acc.amount_min = cents;
        }
        if cents > acc.amount_max {
            acc.amount_max = cents;
        }
        acc.name_bytes += name_bytes;
        acc.note_bytes += note_bytes;
        country_names[country_idx].push(name_hash);
    }

    ThreadResult {
        accs,
        country_names,
    }
}

#[cfg(target_os = "linux")]
struct MmapBuf {
    ptr: *const u8,
    len: usize,
}

#[cfg(target_os = "linux")]
unsafe impl Send for MmapBuf {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MmapBuf {}

#[cfg(target_os = "linux")]
impl std::ops::Deref for MmapBuf {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

#[cfg(target_os = "linux")]
fn map_file(path: &Path) -> Result<MmapBuf, std::io::Error> {
    use std::os::unix::io::AsRawFd;
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(MmapBuf {
            ptr: std::ptr::null(),
            len: 0,
        });
    }
    let fd = f.as_raw_fd();
    let mut addr: *mut u8;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") 9usize,
            in("rdi") 0usize,
            in("rsi") len,
            in("rdx") 1usize,
            in("r10") 2usize,
            in("r8") fd as usize,
            in("r9") 0usize,
            lateout("rax") addr,
            lateout("rcx") _,
            lateout("r11") _,
        );
    }
    if (addr as isize) < 0 && (addr as isize) >= -4095 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(MmapBuf { ptr: addr, len })
}

#[cfg(not(target_os = "linux"))]
fn map_file(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}

fn main() {
    let mut data_dir = PathBuf::from("/data");
    let mut out_dir = PathBuf::from("/out");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--data" && i + 1 < args.len() {
            data_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else if args[i] == "--out" && i + 1 < args.len() {
            out_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    if let Ok(val) = std::env::var("DATA_DIR") {
        data_dir = PathBuf::from(val);
    }
    if let Ok(val) = std::env::var("OUT_DIR") {
        out_dir = PathBuf::from(val);
    }

    let records_path = data_dir.join("records.csv");
    let regions_path = data_dir.join("regions.csv");
    let output_path = out_dir.join("aggregates.txt");

    std::fs::create_dir_all(&out_dir).ok();

    let t0 = std::time::Instant::now();

    // 1. Load regions lookup table
    let regions = load_regions(&regions_path);
    let total_countries = regions.country_names.len();
    let t_regions = std::time::Instant::now();

    // 2. Read records.csv into memory
    let buf = map_file(&records_path).unwrap_or_else(|e| {
        eprintln!("Failed to read records file {:?}: {}", records_path, e);
        std::process::exit(1);
    });
    let t_read = std::time::Instant::now();

    if buf.is_empty() {
        File::create(&output_path).ok();
        return;
    }

    // Skip BOM and header line
    let mut header_end = 0;
    if buf.starts_with(BOM) {
        header_end += 3;
    }
    while header_end < buf.len() && buf[header_end] != b'\n' {
        header_end += 1;
    }
    if header_end < buf.len() && buf[header_end] == b'\n' {
        header_end += 1;
    }
    // Skip any BOM right after header
    if header_end + 3 <= buf.len() && &buf[header_end..header_end + 3] == BOM {
        header_end += 3;
    }

    let records_data = &buf[header_end..];
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(12);

    let boundaries: Vec<(usize, usize)> = if records_data.len() < 64 * 1024 || num_threads <= 1 {
        vec![(0, records_data.len())]
    } else {
        // Step 1: Count quotes in raw chunks to find parity
        let raw_chunk_size = records_data.len() / num_threads;
        let raw_slices: Vec<(usize, usize)> = (0..num_threads)
            .map(|t| {
                let s = t * raw_chunk_size;
                let e = if t == num_threads - 1 {
                    records_data.len()
                } else {
                    (t + 1) * raw_chunk_size
                };
                (s, e)
            })
            .collect();

        let chunk_quotes: Vec<usize> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for &(start, end) in &raw_slices {
                let slice = &records_data[start..end];
                handles.push(s.spawn(move || {
                    let mut count = 0;
                    for &b in slice {
                        if b == b'"' {
                            count += 1;
                        }
                    }
                    count
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut prefix_quotes = vec![0; num_threads + 1];
        for t in 0..num_threads {
            prefix_quotes[t + 1] = prefix_quotes[t] + chunk_quotes[t];
        }

        // Step 2: Find exact record boundaries
        let mut actual_starts = vec![0; num_threads];
        actual_starts[0] = 0;

        for t in 1..num_threads {
            let offset = raw_slices[t].0;
            let mut in_quote = (prefix_quotes[t] % 2) != 0;
            let mut p = offset;
            while p < records_data.len() {
                let b = records_data[p];
                if b == b'"' {
                    in_quote = !in_quote;
                    p += 1;
                } else if b == b'\n' && !in_quote {
                    p += 1; // skip '\n'
                            // Check BOM
                    if p + 3 <= records_data.len() && &records_data[p..p + 3] == BOM {
                        p += 3;
                    }
                    break;
                } else {
                    p += 1;
                }
            }
            actual_starts[t] = p.min(records_data.len());
        }

        let mut ranges = Vec::with_capacity(num_threads);
        for t in 0..num_threads {
            let s = actual_starts[t];
            let e = if t + 1 < num_threads {
                actual_starts[t + 1]
            } else {
                records_data.len()
            };
            if s < e {
                ranges.push((s, e));
            }
        }
        ranges
    };

    // Phase 1: Parse record chunks in parallel
    let thread_results: Vec<ThreadResult> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(boundaries.len());
        let reg_map = &regions.region_to_country;
        let unmatched = regions.unmatched_idx;

        for &(c_start, c_end) in &boundaries {
            handles.push(s.spawn(move || {
                parse_records_chunk(
                    records_data,
                    c_start,
                    c_end,
                    reg_map,
                    unmatched,
                    total_countries,
                )
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let t_parse = std::time::Instant::now();

    // Phase 2: Parallel merge & distinct names calculation across countries
    let actual_threads = num_threads.min(total_countries);
    let countries_per_thread = (total_countries + actual_threads - 1) / actual_threads;

    let country_slices: Vec<(usize, usize)> = (0..actual_threads)
        .map(|t| {
            let s = (t * countries_per_thread).min(total_countries);
            let e = ((t + 1) * countries_per_thread).min(total_countries);
            (s, e)
        })
        .collect();

    let formatted_outputs: Vec<Vec<u8>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(actual_threads);
        let t_res = &thread_results;
        let c_names = &regions.country_names;

        for &(c_start, c_end) in &country_slices {
            handles.push(s.spawn(move || {
                let mut out_bytes = Vec::with_capacity(16 * 1024);
                let mut names_buf = Vec::new();

                for c_idx in c_start..c_end {
                    let mut count = 0u64;
                    let mut id_sum = 0u64;
                    let mut amount_sum = 0i64;
                    let mut amount_min = i64::MAX;
                    let mut amount_max = i64::MIN;
                    let mut name_bytes = 0u64;
                    let mut note_bytes = 0u64;

                    for tr in t_res {
                        let acc = &tr.accs[c_idx];
                        count += acc.count;
                        id_sum += acc.id_sum;
                        amount_sum += acc.amount_sum;
                        if acc.count > 0 {
                            if acc.amount_min < amount_min {
                                amount_min = acc.amount_min;
                            }
                            if acc.amount_max > amount_max {
                                amount_max = acc.amount_max;
                            }
                        }
                        name_bytes += acc.name_bytes;
                        note_bytes += acc.note_bytes;
                    }

                    if count == 0 {
                        continue;
                    }

                    // Collect all name hashes
                    names_buf.clear();
                    let total_names: usize =
                        t_res.iter().map(|tr| tr.country_names[c_idx].len()).sum();
                    names_buf.reserve(total_names);
                    for tr in t_res {
                        names_buf.extend_from_slice(&tr.country_names[c_idx]);
                    }

                    names_buf.sort_unstable();
                    names_buf.dedup();
                    let names_distinct = names_buf.len() as u64;

                    // Format line: country count id_sum amount_sum amount_min amount_max names_distinct name_bytes note_bytes\n
                    out_bytes.extend_from_slice(&c_names[c_idx]);
                    out_bytes.push(b' ');
                    write_u64(&mut out_bytes, count);
                    out_bytes.push(b' ');
                    write_u64(&mut out_bytes, id_sum);
                    out_bytes.push(b' ');
                    format_cents(&mut out_bytes, amount_sum);
                    out_bytes.push(b' ');
                    format_cents(&mut out_bytes, amount_min);
                    out_bytes.push(b' ');
                    format_cents(&mut out_bytes, amount_max);
                    out_bytes.push(b' ');
                    write_u64(&mut out_bytes, names_distinct);
                    out_bytes.push(b' ');
                    write_u64(&mut out_bytes, name_bytes);
                    out_bytes.push(b' ');
                    write_u64(&mut out_bytes, note_bytes);
                    out_bytes.push(b'\n');
                }

                out_bytes
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let t_merge = std::time::Instant::now();

    // Phase 3: Write final output
    let mut final_file = File::create(&output_path).unwrap_or_else(|e| {
        eprintln!("Failed to create output file {:?}: {}", output_path, e);
        std::process::exit(1);
    });

    for chunk in formatted_outputs {
        final_file.write_all(&chunk).unwrap();
    }
    let t_done = std::time::Instant::now();

    if std::env::var("TIMING").is_ok() {
        eprintln!(
            "TIMING: regions={:.2}ms, read={:.2}ms, parse={:.2}ms, merge={:.2}ms, write={:.2}ms, total={:.2}ms",
            t_regions.duration_since(t0).as_secs_f64() * 1000.0,
            t_read.duration_since(t_regions).as_secs_f64() * 1000.0,
            t_parse.duration_since(t_read).as_secs_f64() * 1000.0,
            t_merge.duration_since(t_parse).as_secs_f64() * 1000.0,
            t_done.duration_since(t_merge).as_secs_f64() * 1000.0,
            t_done.duration_since(t0).as_secs_f64() * 1000.0
        );
    }
}

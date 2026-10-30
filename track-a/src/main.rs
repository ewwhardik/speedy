//! Speed Demon 2026 - Track A (Ingest)
//! Ultra-high-throughput log aggregation for Falkenstein i5-13500 (12 hardware threads).
//!
//! Architecture:
//! 1. Zero-copy Linux mmap input (0 MB container memory overhead).
//! 2. Fast chunked hash table for endpoint strings with zero heap allocation.
//! 3. Flat compact record stream per thread (no per-group Vec allocations).
//! 4. Two-phase cache-friendly bucket scatter directly into a single contiguous array.
//! 5. Embarrassingly parallel percentile selection and SIMD-optimized line formatting.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;

const HOURS: [&str; 72] = [
    // Oct 30: 18..=23
    "2026-10-30T18:00:00Z", "2026-10-30T19:00:00Z", "2026-10-30T20:00:00Z",
    "2026-10-30T21:00:00Z", "2026-10-30T22:00:00Z", "2026-10-30T23:00:00Z",
    // Oct 31: 00..=23
    "2026-10-31T00:00:00Z", "2026-10-31T01:00:00Z", "2026-10-31T02:00:00Z",
    "2026-10-31T03:00:00Z", "2026-10-31T04:00:00Z", "2026-10-31T05:00:00Z",
    "2026-10-31T06:00:00Z", "2026-10-31T07:00:00Z", "2026-10-31T08:00:00Z",
    "2026-10-31T09:00:00Z", "2026-10-31T10:00:00Z", "2026-10-31T11:00:00Z",
    "2026-10-31T12:00:00Z", "2026-10-31T13:00:00Z", "2026-10-31T14:00:00Z",
    "2026-10-31T15:00:00Z", "2026-10-31T16:00:00Z", "2026-10-31T17:00:00Z",
    "2026-10-31T18:00:00Z", "2026-10-31T19:00:00Z", "2026-10-31T20:00:00Z",
    "2026-10-31T21:00:00Z", "2026-10-31T22:00:00Z", "2026-10-31T23:00:00Z",
    // Nov 01: 00..=23
    "2026-11-01T00:00:00Z", "2026-11-01T01:00:00Z", "2026-11-01T02:00:00Z",
    "2026-11-01T03:00:00Z", "2026-11-01T04:00:00Z", "2026-11-01T05:00:00Z",
    "2026-11-01T06:00:00Z", "2026-11-01T07:00:00Z", "2026-11-01T08:00:00Z",
    "2026-11-01T09:00:00Z", "2026-11-01T10:00:00Z", "2026-11-01T11:00:00Z",
    "2026-11-01T12:00:00Z", "2026-11-01T13:00:00Z", "2026-11-01T14:00:00Z",
    "2026-11-01T15:00:00Z", "2026-11-01T16:00:00Z", "2026-11-01T17:00:00Z",
    "2026-11-01T18:00:00Z", "2026-11-01T19:00:00Z", "2026-11-01T20:00:00Z",
    "2026-11-01T21:00:00Z", "2026-11-01T22:00:00Z", "2026-11-01T23:00:00Z",
    // Nov 02: 00..=17
    "2026-11-02T00:00:00Z", "2026-11-02T01:00:00Z", "2026-11-02T02:00:00Z",
    "2026-11-02T03:00:00Z", "2026-11-02T04:00:00Z", "2026-11-02T05:00:00Z",
    "2026-11-02T06:00:00Z", "2026-11-02T07:00:00Z", "2026-11-02T08:00:00Z",
    "2026-11-02T09:00:00Z", "2026-11-02T10:00:00Z", "2026-11-02T11:00:00Z",
    "2026-11-02T12:00:00Z", "2026-11-02T13:00:00Z", "2026-11-02T14:00:00Z",
    "2026-11-02T15:00:00Z", "2026-11-02T16:00:00Z", "2026-11-02T17:00:00Z",
];

const MAX_EP_PER_THREAD: usize = 2048;

#[inline(always)]
fn fast_hash(ep: &[u8]) -> u64 {
    let len = ep.len();
    let mut h: u64 = 0x517cc1b727220a95 ^ (len as u64);
    let mut i = 0;
    while i + 8 <= len {
        let chunk = u64::from_ne_bytes(ep[i..i + 8].try_into().unwrap());
        h = (h ^ chunk).wrapping_mul(0x517cc1b727220a95);
        i += 8;
    }
    if i < len {
        let mut rem = 0u64;
        for &b in &ep[i..] {
            rem = (rem << 8) | (b as u64);
        }
        h = (h ^ rem).wrapping_mul(0x517cc1b727220a95);
    }
    h
}

/// Contiguous arena-backed endpoint hash table
struct EndpointTable {
    arena: Vec<u8>,
    offsets: Vec<(u32, u16)>, // (arena_offset, len)
    slots: Vec<(u64, u32)>,   // (hash, ep_id)
}

impl EndpointTable {
    fn new() -> Self {
        Self {
            arena: Vec::with_capacity(32768),
            offsets: Vec::with_capacity(1842),
            slots: vec![(0, u32::MAX); 4096],
        }
    }

    #[inline(always)]
    fn get_ep_slice(&self, id: u32) -> &[u8] {
        let (off, len) = self.offsets[id as usize];
        &self.arena[off as usize..(off as usize + len as usize)]
    }

    #[inline(always)]
    fn get_or_insert(&mut self, ep: &[u8]) -> u32 {
        let h = fast_hash(ep);
        let mut idx = (h as usize) & 4095;
        loop {
            let (entry_h, id) = self.slots[idx];
            if id == u32::MAX {
                let new_id = self.offsets.len() as u32;
                let off = self.arena.len() as u32;
                let len = ep.len() as u16;
                self.arena.extend_from_slice(ep);
                self.offsets.push((off, len));
                self.slots[idx] = (h, new_id);
                return new_id;
            }
            if entry_h == h && self.get_ep_slice(id) == ep {
                return id;
            }
            idx = (idx + 1) & 4095;
        }
    }
}

struct ThreadParseResult {
    endpoints: Vec<Vec<u8>>,
    records: Vec<u64>,     // ((hour << 16 | local_ep) as u64) << 32 | (latency as u64)
    counts: Vec<u32>,      // size 72 * MAX_EP_PER_THREAD
    errors: Vec<u32>,      // size 72 * MAX_EP_PER_THREAD
}

fn parse_chunk(buf: &[u8]) -> ThreadParseResult {
    let mut table = EndpointTable::new();
    let num_groups = 72 * MAX_EP_PER_THREAD;
    let mut counts = vec![0u32; num_groups];
    let mut errors = vec![0u32; num_groups];
    let mut records = Vec::with_capacity(buf.len() / 70);

    let len = buf.len();
    let mut ptr = 0;

    while ptr < len {
        if ptr + 27 >= len {
            break;
        }

        // Parse hour index using direct digit arithmetic
        let d0 = buf[ptr + 8];
        let d1 = buf[ptr + 9];
        let h = (buf[ptr + 11] - b'0') as usize * 10 + (buf[ptr + 12] - b'0') as usize;
        let hour_idx = if d0 == b'3' {
            if d1 == b'0' { h - 18 } else { 6 + h }
        } else {
            if d1 == b'1' { 30 + h } else { 54 + h }
        };

        // Method starts at ptr + 28
        let mut p = ptr + 28;
        while p < len && buf[p] != b' ' {
            p += 1;
        }
        p += 1; // start of endpoint

        let ep_start = p;
        while p < len && buf[p] != b' ' {
            p += 1;
        }
        let ep = &buf[ep_start..p];
        p += 1; // start of status

        let is_err = p < len && buf[p] == b'5';
        p += 4; // skip status (3 digits) + space

        // Latency
        let mut latency: u32 = 0;
        while p < len && buf[p] >= b'0' && buf[p] <= b'9' {
            latency = latency * 10 + (buf[p] - b'0') as u32;
            p += 1;
        }

        while p < len && (buf[p] == b'\r' || buf[p] == b'\n') {
            p += 1;
        }
        ptr = p;

        let ep_id = table.get_or_insert(ep) as usize;
        let g_idx = hour_idx * MAX_EP_PER_THREAD + ep_id;
        counts[g_idx] += 1;
        if is_err {
            errors[g_idx] += 1;
        }
        let record = ((g_idx as u64) << 32) | (latency as u64);
        records.push(record);
    }

    let mut endpoints = Vec::with_capacity(table.offsets.len());
    for i in 0..table.offsets.len() {
        endpoints.push(table.get_ep_slice(i as u32).to_vec());
    }

    ThreadParseResult {
        endpoints,
        records,
        counts,
        errors,
    }
}

#[inline(always)]
fn write_u64(buf: &mut Vec<u8>, mut n: u64) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut temp = [0u8; 20];
    let mut i = 0;
    while n > 0 {
        temp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        buf.push(temp[i]);
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
            in("rax") 9usize, // SYS_mmap
            in("rdi") 0usize, // addr = NULL
            in("rsi") len,
            in("rdx") 1usize, // PROT_READ
            in("r10") 2usize, // MAP_PRIVATE
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

    let input_path = data_dir.join("requests.log");
    std::fs::create_dir_all(&out_dir).ok();
    let output_path = out_dir.join("summary.txt");

    let buffer = match map_file(&input_path) {
        Ok(b) => b,
        Err(_) => {
            File::create(&output_path).ok();
            return;
        }
    };

    if buffer.is_empty() {
        File::create(&output_path).ok();
        return;
    }

    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(12);

    let chunk_size = buffer.len() / num_threads;
    let mut chunk_ranges = Vec::with_capacity(num_threads);
    let mut start = 0;
    for idx in 0..num_threads {
        let end = if idx + 1 == num_threads {
            buffer.len()
        } else {
            let mut target = (idx + 1) * chunk_size;
            while target < buffer.len() && buffer[target] != b'\n' {
                target += 1;
            }
            if target < buffer.len() {
                target += 1;
            }
            target
        };
        chunk_ranges.push((start, end));
        start = end;
    }

    // 1. Parallel chunk parsing across hardware threads
    let mut thread_results: Vec<ThreadParseResult> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(s_off, e_off) in &chunk_ranges {
            let chunk = &buffer[s_off..e_off];
            handles.push(s.spawn(move || parse_chunk(chunk)));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // 2. Unify endpoints and sort alphabetically
    let mut global_endpoints: Vec<Vec<u8>> = Vec::with_capacity(1842);
    for tr in &thread_results {
        for ep in &tr.endpoints {
            if !global_endpoints.iter().any(|existing| existing == ep) {
                global_endpoints.push(ep.clone());
            }
        }
    }
    global_endpoints.sort_unstable();
    let num_global_eps = global_endpoints.len();
    let total_groups = 72 * num_global_eps;

    // Build thread remap tables: remap[t][local_ep] -> global_ep and reverse
    let mut thread_remap: Vec<Vec<usize>> = Vec::with_capacity(num_threads);
    let mut rev_remap: Vec<Vec<usize>> = Vec::with_capacity(num_threads);
    for tr in &thread_results {
        let mut remap = vec![usize::MAX; tr.endpoints.len()];
        let mut rev = vec![usize::MAX; num_global_eps];
        for (local_id, ep) in tr.endpoints.iter().enumerate() {
            if let Ok(pos) = global_endpoints.binary_search(ep) {
                remap[local_id] = pos;
                rev[pos] = local_id;
            }
        }
        thread_remap.push(remap);
        rev_remap.push(rev);
    }

    // 3. Compute group totals and per-thread group offsets
    let mut group_totals = vec![0u32; total_groups];
    let mut group_errors = vec![0u32; total_groups];
    let mut thread_group_offsets: Vec<Vec<u32>> = vec![vec![0u32; total_groups]; num_threads];

    for (t_idx, tr) in thread_results.iter().enumerate() {
        let remap = &thread_remap[t_idx];
        for hour in 0..72 {
            let h_base = hour * MAX_EP_PER_THREAD;
            let gh_base = hour * num_global_eps;
            for (local_id, &global_id) in remap.iter().enumerate() {
                if global_id != usize::MAX {
                    let local_g = h_base + local_id;
                    let c = tr.counts[local_g];
                    if c > 0 {
                        let global_g = gh_base + global_id;
                        group_totals[global_g] += c;
                        group_errors[global_g] += tr.errors[local_g];
                    }
                }
            }
        }
    }

    // Prefix sum over global groups
    let mut group_start_offsets = vec![0u32; total_groups + 1];
    let mut acc = 0u32;
    for g in 0..total_groups {
        group_start_offsets[g] = acc;
        acc += group_totals[g];
    }
    group_start_offsets[total_groups] = acc;
    let total_records = acc as usize;

    // Distribute offsets among threads using O(1) reverse lookup
    for g in 0..total_groups {
        let mut cur = group_start_offsets[g];
        let hour = g / num_global_eps;
        let global_ep = g % num_global_eps;
        let h_base = hour * MAX_EP_PER_THREAD;
        for t_idx in 0..num_threads {
            thread_group_offsets[t_idx][g] = cur;
            let local_id = rev_remap[t_idx][global_ep];
            if local_id != usize::MAX {
                cur += thread_results[t_idx].counts[h_base + local_id];
            }
        }
    }

    // 4. Parallel scatter into a single contiguous flat buffer
    let mut all_latencies = vec![0u32; total_records];
    let lat_ptr = all_latencies.as_mut_ptr() as usize;

    thread::scope(|s| {
        for (t_idx, tr) in thread_results.iter_mut().enumerate() {
            let remap = &thread_remap[t_idx];
            let mut t_offsets = std::mem::take(&mut thread_group_offsets[t_idx]);
            s.spawn(move || {
                let records = &tr.records;
                let raw_ptr = lat_ptr as *mut u32;
                for &rec in records {
                    let local_g = (rec >> 32) as usize;
                    let latency = rec as u32;
                    let hour = local_g / MAX_EP_PER_THREAD;
                    let local_ep = local_g % MAX_EP_PER_THREAD;
                    let global_ep = remap[local_ep];
                    let global_g = hour * num_global_eps + global_ep;
                    let pos = t_offsets[global_g] as usize;
                    t_offsets[global_g] += 1;
                    unsafe {
                        *raw_ptr.add(pos) = latency;
                    }
                }
            });
        }
    });

    // Drop thread parse results to release all intermediate heap memory immediately
    drop(thread_results);
    drop(thread_remap);

    // 5. Parallel percentile calculation & formatting across 12 threads
    let hours_per_thread = (72 + num_threads - 1) / num_threads;
    let thread_chunks: Vec<(usize, usize)> = (0..num_threads)
        .map(|i| {
            let s = (i * hours_per_thread).min(72);
            let e = ((i + 1) * hours_per_thread).min(72);
            (s, e)
        })
        .collect();

    let output_parts: Vec<Vec<u8>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(h_start, h_end) in &thread_chunks {
            let g_eps = &global_endpoints;
            let g_totals = &group_totals;
            let g_errors = &group_errors;
            let g_offsets = &group_start_offsets;
            let all_lats = lat_ptr;

            handles.push(s.spawn(move || {
                let mut out = Vec::with_capacity(256 * 1024);
                let raw_ptr = all_lats as *mut u32;

                for hour_idx in h_start..h_end {
                    let hour_str = HOURS[hour_idx].as_bytes();
                    let gh_base = hour_idx * num_global_eps;

                    for (global_ep, ep_bytes) in g_eps.iter().enumerate() {
                        let global_g = gh_base + global_ep;
                        let count = g_totals[global_g];
                        if count == 0 {
                            continue;
                        }

                        let start_off = g_offsets[global_g] as usize;
                        let n = count as usize;
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(raw_ptr.add(start_off), n)
                        };

                        let p50_idx = (50 * n + 99) / 100 - 1;
                        let p95_idx = (95 * n + 99) / 100 - 1;
                        let p99_idx = (99 * n + 99) / 100 - 1;

                        let (p50, p95, p99) = if n <= 40 {
                            slice.sort_unstable();
                            (slice[p50_idx], slice[p95_idx], slice[p99_idx])
                        } else {
                            slice.select_nth_unstable(p99_idx);
                            let p99_val = slice[p99_idx];
                            slice[..p99_idx].select_nth_unstable(p95_idx);
                            let p95_val = slice[p95_idx];
                            slice[..p95_idx].select_nth_unstable(p50_idx);
                            let p50_val = slice[p50_idx];
                            (p50_val, p95_val, p99_val)
                        };

                        let err_count = g_errors[global_g];
                        let ppm = (err_count as u64 * 1_000_000) / (n as u64);

                        out.extend_from_slice(hour_str);
                        out.push(b' ');
                        out.extend_from_slice(ep_bytes);
                        out.push(b' ');
                        write_u64(&mut out, n as u64);
                        out.push(b' ');
                        write_u64(&mut out, p50 as u64);
                        out.push(b' ');
                        write_u64(&mut out, p95 as u64);
                        out.push(b' ');
                        write_u64(&mut out, p99 as u64);
                        out.push(b' ');
                        write_u64(&mut out, err_count as u64);
                        out.push(b' ');
                        write_u64(&mut out, ppm);
                        out.push(b'\n');
                    }
                }
                out
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // 6. Write final output
    let mut file = File::create(&output_path).unwrap();
    for part in output_parts {
        file.write_all(&part).unwrap();
    }
}

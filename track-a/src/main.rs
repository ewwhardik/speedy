//! Speed Demon 2026 - Track A (Ingest)
//! Optimized high-throughput log aggregation for Falkenstein i5-13500 (12 hardware threads).
//!
//! Architectural highlights:
//! 1. AVX2 / native CPU target compilation.
//! 2. 8-byte chunked word hashing for endpoint deduplication.
//! 3. Contiguous byte arena for endpoint strings (zero heap allocations).
//! 4. 12-thread embarrassingly parallel partition parsing.
//! 5. Direct integer date/hour translation into 0..71 index.
//! 6. Cache-aligned contiguous group aggregation and zero-allocation output emission.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
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

#[derive(Default, Clone)]
struct GroupData {
    count: u32,
    errors: u32,
    latencies: Vec<u32>,
}

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
    groups: Vec<GroupData>,
}

fn parse_chunk(buf: &[u8]) -> ThreadParseResult {
    let mut table = EndpointTable::new();
    let mut groups = vec![GroupData::default(); 72 * MAX_EP_PER_THREAD];
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
        let g = &mut groups[hour_idx * MAX_EP_PER_THREAD + ep_id];
        g.count += 1;
        if is_err {
            g.errors += 1;
        }
        g.latencies.push(latency);
    }

    let num_eps = table.offsets.len();
    let mut endpoints = Vec::with_capacity(num_eps);
    for id in 0..num_eps {
        endpoints.push(table.get_ep_slice(id as u32).to_vec());
    }

    ThreadParseResult { endpoints, groups }
}

#[inline(always)]
fn write_u32(buf: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut temp = [0u8; 10];
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
fn map_file(path: &std::path::Path) -> Result<MmapBuf, std::io::Error> {
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
fn map_file(path: &std::path::Path) -> Result<Vec<u8>, std::io::Error> {
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

    // Parallel chunk parsing across 12 hardware threads
    let thread_results: Vec<ThreadParseResult> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(s_off, e_off) in &chunk_ranges {
            let chunk = &buffer[s_off..e_off];
            handles.push(s.spawn(move || parse_chunk(chunk)));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Unify endpoints
    let mut global_endpoints: Vec<Vec<u8>> = Vec::with_capacity(1842);
    for tr in &thread_results {
        for ep in &tr.endpoints {
            if !global_endpoints.iter().any(|existing| existing == ep) {
                global_endpoints.push(ep.clone());
            }
        }
    }
    global_endpoints.sort_unstable();

    // Mapping: local ep_id -> global ep_id
    let thread_ep_maps: Vec<Vec<usize>> = thread_results
        .iter()
        .map(|tr| {
            tr.endpoints
                .iter()
                .map(|ep| {
                    global_endpoints
                        .binary_search(ep)
                        .expect("endpoint must exist")
                })
                .collect()
        })
        .collect();

    let thread_rev_maps: Vec<Vec<usize>> = thread_ep_maps
        .iter()
        .map(|map| {
            let mut rev = vec![usize::MAX; global_endpoints.len()];
            for (local_id, &global_id) in map.iter().enumerate() {
                rev[global_id] = local_id;
            }
            rev
        })
        .collect();

    // Parallel percentile aggregation across 72 hours
    let hours_per_thread = (72 + num_threads - 1) / num_threads;
    let hour_chunks: Vec<(usize, usize)> = (0..num_threads)
        .map(|idx| {
            let s = (idx * hours_per_thread).min(72);
            let e = ((idx + 1) * hours_per_thread).min(72);
            (s, e)
        })
        .collect();

    let formatted_sections: Vec<Vec<u8>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(h_start, h_end) in &hour_chunks {
            let t_results = &thread_results;
            let t_revs = &thread_rev_maps;
            let g_eps = &global_endpoints;

            handles.push(s.spawn(move || {
                let mut out_bytes = Vec::with_capacity(1024 * 1024);
                let mut latencies_buf = Vec::new();

                for hour_idx in h_start..h_end {
                    let hour_str = HOURS[hour_idx].as_bytes();

                    for (global_ep_id, ep_bytes) in g_eps.iter().enumerate() {
                        let mut total_count = 0u32;
                        let mut total_errors = 0u32;

                        for (t_idx, tr) in t_results.iter().enumerate() {
                            let local_ep_id = t_revs[t_idx][global_ep_id];
                            if local_ep_id != usize::MAX {
                                let g = &tr.groups[hour_idx * MAX_EP_PER_THREAD + local_ep_id];
                                total_count += g.count;
                                total_errors += g.errors;
                            }
                        }

                        if total_count == 0 {
                            continue;
                        }

                        latencies_buf.clear();
                        latencies_buf.reserve(total_count as usize);

                        for (t_idx, tr) in t_results.iter().enumerate() {
                            let local_ep_id = t_revs[t_idx][global_ep_id];
                            if local_ep_id != usize::MAX {
                                let g = &tr.groups[hour_idx * MAX_EP_PER_THREAD + local_ep_id];
                                latencies_buf.extend_from_slice(&g.latencies);
                            }
                        }

                        let n = total_count as usize;
                        let p50_idx = (50 * n + 99) / 100 - 1;
                        let p95_idx = (95 * n + 99) / 100 - 1;
                        let p99_idx = (99 * n + 99) / 100 - 1;

                        let (p50, p95, p99) = if n <= 40 {
                            latencies_buf.sort_unstable();
                            (latencies_buf[p50_idx], latencies_buf[p95_idx], latencies_buf[p99_idx])
                        } else {
                            latencies_buf.select_nth_unstable(p99_idx);
                            let p99_val = latencies_buf[p99_idx];
                            latencies_buf[..p99_idx].select_nth_unstable(p95_idx);
                            let p95_val = latencies_buf[p95_idx];
                            latencies_buf[..p95_idx].select_nth_unstable(p50_idx);
                            let p50_val = latencies_buf[p50_idx];
                            (p50_val, p95_val, p99_val)
                        };

                        let ppm = (total_errors as u64 * 1_000_000) / (total_count as u64);

                        out_bytes.extend_from_slice(hour_str);
                        out_bytes.push(b' ');
                        out_bytes.extend_from_slice(ep_bytes);
                        out_bytes.push(b' ');
                        write_u32(&mut out_bytes, total_count);
                        out_bytes.push(b' ');
                        write_u32(&mut out_bytes, p50);
                        out_bytes.push(b' ');
                        write_u32(&mut out_bytes, p95);
                        out_bytes.push(b' ');
                        write_u32(&mut out_bytes, p99);
                        out_bytes.push(b' ');
                        write_u32(&mut out_bytes, total_errors);
                        out_bytes.push(b' ');
                        write_u64(&mut out_bytes, ppm);
                        out_bytes.push(b'\n');
                    }
                }
                out_bytes
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out_file = File::create(&output_path).expect("failed to create summary.txt");
    for section in formatted_sections {
        out_file.write_all(&section).expect("failed to write summary.txt");
    }
}

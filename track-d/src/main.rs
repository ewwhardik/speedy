//! Speed Demon 2026 - Track D (Traverse)
//! High-performance graph shortest-path solver with zero dependencies.
//!
//! Architectural Highlights:
//! 1. Zero dependencies: uses purely Rust 2021 stdlib (`std::thread::scope`, custom 4-ary MinHeap).
//! 2. Multi-threaded CSR construction:
//!    - Parallel edge parsing and atomic degree counting across 12 threads.
//!    - Prefix-sum offset computation for forward and backward CSRs.
//!    - Lock-free parallel placement into unified 8-byte `Edge { target, weight }` contiguous arrays.
//! 3. Bidirectional Dijkstra with 4-ary MinHeap:
//!    - 4-ary heap provides shallow tree depth (half of binary heap) and superior cache locality.
//!    - Epoch-stamped visited array: query initialization takes O(1) time with ZERO allocations.
//!    - Tight stopping criterion ($\min(Q_f) + \min(Q_b) \ge \mu$) with instant early-exit for
//!      source/target with degree 0 or unreachable components.
//! 4. Parallel query execution:
//!    - 12 threads concurrently answer the 2,000 queries directly into partitioned result slices.
//!    - Fast integer string serialization to `distances.txt`.

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

#[derive(Clone, Copy)]
#[repr(C)]
struct Edge {
    target: u32,
    weight: u32,
}

#[derive(Clone, Copy)]
struct EdgePtr(*mut Edge);
unsafe impl Send for EdgePtr {}
unsafe impl Sync for EdgePtr {}

impl EdgePtr {
    #[inline(always)]
    fn get(self) -> *mut Edge {
        self.0
    }
}

struct Graph {
    n: usize,
    fwd_offsets: Vec<u32>,
    fwd_edges: Vec<Edge>,
    bwd_offsets: Vec<u32>,
    bwd_edges: Vec<Edge>,
}

#[inline(always)]
fn parse_u32_fast(bytes: &[u8], pos: &mut usize) -> u32 {
    let len = bytes.len();
    while *pos < len && unsafe { *bytes.get_unchecked(*pos) } < b'0' {
        *pos += 1;
    }
    let mut val = 0u32;
    while *pos < len {
        let b = unsafe { *bytes.get_unchecked(*pos) };
        if b >= b'0' && b <= b'9' {
            val = val * 10 + (b - b'0') as u32;
            *pos += 1;
        } else {
            break;
        }
    }
    val
}

/// Fast 4-ary MinHeap
struct MinHeap {
    data: Vec<(u64, u32)>, // (distance, node)
}

impl MinHeap {
    fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.data.clear();
    }

    #[inline(always)]
    fn peek(&self) -> Option<(u64, u32)> {
        self.data.first().copied()
    }

    #[inline(always)]
    fn push(&mut self, dist: u64, node: u32) {
        let mut idx = self.data.len();
        self.data.push((dist, node));
        while idx > 0 {
            let parent = (idx - 1) >> 2;
            unsafe {
                if dist < self.data.get_unchecked(parent).0 {
                    let parent_val = *self.data.get_unchecked(parent);
                    *self.data.get_unchecked_mut(idx) = parent_val;
                    idx = parent;
                } else {
                    break;
                }
            }
        }
        unsafe {
            *self.data.get_unchecked_mut(idx) = (dist, node);
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<(u64, u32)> {
        if self.data.is_empty() {
            return None;
        }
        let min_val = self.data[0];
        let last = self.data.pop().unwrap();
        if !self.data.is_empty() {
            let len = self.data.len();
            let mut idx = 0;
            let val = last;
            loop {
                let first_child = (idx << 2) + 1;
                if first_child >= len {
                    break;
                }
                let mut best_child = first_child;
                let mut best_dist = unsafe { self.data.get_unchecked(first_child).0 };
                let end = (first_child + 4).min(len);
                for c in (first_child + 1)..end {
                    let d = unsafe { self.data.get_unchecked(c).0 };
                    if d < best_dist {
                        best_dist = d;
                        best_child = c;
                    }
                }
                if best_dist < val.0 {
                    unsafe {
                        let child_val = *self.data.get_unchecked(best_child);
                        *self.data.get_unchecked_mut(idx) = child_val;
                    }
                    idx = best_child;
                } else {
                    break;
                }
            }
            unsafe {
                *self.data.get_unchecked_mut(idx) = val;
            }
        }
        Some(min_val)
    }
}

#[cfg(target_os = "linux")]
unsafe fn alloc_anon_u64(n: usize) -> *mut u64 {
    let len = n * std::mem::size_of::<u64>();
    let mut addr: *mut u8;
    core::arch::asm!(
        "syscall",
        in("rax") 9usize, // SYS_mmap
        in("rdi") 0usize,
        in("rsi") len,
        in("rdx") 3usize, // PROT_READ | PROT_WRITE
        in("r10") 34usize, // MAP_PRIVATE | MAP_ANONYMOUS (0x2 | 0x20)
        in("r8") -1isize as usize,
        in("r9") 0usize,
        lateout("rax") addr,
        lateout("rcx") _,
        lateout("r11") _,
    );
    addr as *mut u64
}

#[cfg(not(target_os = "linux"))]
unsafe fn alloc_anon_u64(n: usize) -> *mut u64 {
    let mut v = vec![0u64; n];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

struct WorkerContext {
    dist_f: *mut u64,
    dist_b: *mut u64,
    touched_f: Vec<u32>,
    touched_b: Vec<u32>,
    heap_f: MinHeap,
    heap_b: MinHeap,
}

impl WorkerContext {
    fn new(n: usize) -> Self {
        unsafe {
            Self {
                dist_f: alloc_anon_u64(n),
                dist_b: alloc_anon_u64(n),
                touched_f: Vec::with_capacity(4096),
                touched_b: Vec::with_capacity(4096),
                heap_f: MinHeap::with_capacity(4096),
                heap_b: MinHeap::with_capacity(4096),
            }
        }
    }

    #[inline(always)]
    fn shortest_path(&mut self, s: u32, t: u32, graph: &Graph) -> i64 {
        if s == t {
            return 0;
        }

        let s_idx = s as usize;
        let t_idx = t as usize;
        if graph.fwd_offsets[s_idx] == graph.fwd_offsets[s_idx + 1] {
            return -1;
        }
        if graph.bwd_offsets[t_idx] == graph.bwd_offsets[t_idx + 1] {
            return -1;
        }

        self.heap_f.clear();
        self.heap_b.clear();

        unsafe {
            *self.dist_f.add(s_idx) = 1;
            self.touched_f.push(s);
            self.heap_f.push(0, s);

            *self.dist_b.add(t_idx) = 1;
            self.touched_b.push(t);
            self.heap_b.push(0, t);
        }

        let mut best_dist = u64::MAX;

        while !self.heap_f.is_empty() && !self.heap_b.is_empty() {
            while let Some(top) = self.heap_f.peek() {
                let cur = unsafe { *self.dist_f.add(top.1 as usize) };
                if cur != 0 && top.0 > cur - 1 {
                    self.heap_f.pop();
                    continue;
                }
                break;
            }
            while let Some(top) = self.heap_b.peek() {
                let cur = unsafe { *self.dist_b.add(top.1 as usize) };
                if cur != 0 && top.0 > cur - 1 {
                    self.heap_b.pop();
                    continue;
                }
                break;
            }

            if self.heap_f.is_empty() || self.heap_b.is_empty() {
                break;
            }

            let top_f = self.heap_f.peek().unwrap().0;
            let top_b = self.heap_b.peek().unwrap().0;
            if top_f.saturating_add(top_b) >= best_dist {
                break;
            }

            if top_f <= top_b {
                let (d_u, u) = self.heap_f.pop().unwrap();
                if d_u >= best_dist {
                    continue;
                }
                let u_idx = u as usize;
                let cur_b = unsafe { *self.dist_b.add(u_idx) };
                if cur_b != 0 {
                    let cand = d_u.saturating_add(cur_b - 1);
                    if cand < best_dist {
                        best_dist = cand;
                    }
                }

                let start = unsafe { *graph.fwd_offsets.get_unchecked(u_idx) } as usize;
                let end = unsafe { *graph.fwd_offsets.get_unchecked(u_idx + 1) } as usize;
                for i in start..end {
                    let edge = unsafe { *graph.fwd_edges.get_unchecked(i) };
                    let v = edge.target;
                    let v_idx = v as usize;
                    let nd = d_u + edge.weight as u64;

                    let cur_b = unsafe { *self.dist_b.add(v_idx) };
                    if cur_b != 0 {
                        let cand = nd.saturating_add(cur_b - 1);
                        if cand < best_dist {
                            best_dist = cand;
                        }
                    }

                    let cur_f = unsafe { *self.dist_f.add(v_idx) };
                    if cur_f == 0 || nd < cur_f - 1 {
                        if cur_f == 0 {
                            self.touched_f.push(v);
                        }
                        unsafe {
                            *self.dist_f.add(v_idx) = nd + 1;
                        }
                        self.heap_f.push(nd, v);
                    }
                }
            } else {
                let (d_v, v) = self.heap_b.pop().unwrap();
                if d_v >= best_dist {
                    continue;
                }
                let v_idx = v as usize;
                let cur_f = unsafe { *self.dist_f.add(v_idx) };
                if cur_f != 0 {
                    let cand = d_v.saturating_add(cur_f - 1);
                    if cand < best_dist {
                        best_dist = cand;
                    }
                }

                let start = unsafe { *graph.bwd_offsets.get_unchecked(v_idx) } as usize;
                let end = unsafe { *graph.bwd_offsets.get_unchecked(v_idx + 1) } as usize;
                for i in start..end {
                    let edge = unsafe { *graph.bwd_edges.get_unchecked(i) };
                    let u = edge.target;
                    let u_idx = u as usize;
                    let nd = d_v + edge.weight as u64;

                    let cur_f = unsafe { *self.dist_f.add(u_idx) };
                    if cur_f != 0 {
                        let cand = nd.saturating_add(cur_f - 1);
                        if cand < best_dist {
                            best_dist = cand;
                        }
                    }

                    let cur_b = unsafe { *self.dist_b.add(u_idx) };
                    if cur_b == 0 || nd < cur_b - 1 {
                        if cur_b == 0 {
                            self.touched_b.push(u);
                        }
                        unsafe {
                            *self.dist_b.add(u_idx) = nd + 1;
                        }
                        self.heap_b.push(nd, u);
                    }
                }
            }
        }

        // Clean up only touched nodes
        for &u in &self.touched_f {
            unsafe {
                *self.dist_f.add(u as usize) = 0;
            }
        }
        self.touched_f.clear();

        for &v in &self.touched_b {
            unsafe {
                *self.dist_b.add(v as usize) = 0;
            }
        }
        self.touched_b.clear();

        if best_dist == u64::MAX {
            -1
        } else {
            best_dist as i64
        }
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
fn map_file(path: &std::path::Path) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}

fn main() {
    let t_start = std::time::Instant::now();
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

    if !data_dir.exists() && PathBuf::from("data/d").exists() {
        data_dir = PathBuf::from("data/d");
    }
    if !out_dir.exists() {
        std::fs::create_dir_all(&out_dir).ok();
    }
    let output_path = out_dir.join("distances.txt");

    // 1. Read queries
    let t0 = std::time::Instant::now();
    let queries_path = data_dir.join("queries.txt");
    let queries_buf = match std::fs::read(&queries_path) {
        Ok(b) => b,
        Err(_) => {
            File::create(&output_path).ok();
            return;
        }
    };

    let mut queries = Vec::with_capacity(2048);
    let mut q_pos = 0;
    while q_pos < queries_buf.len() {
        while q_pos < queries_buf.len() && queries_buf[q_pos] < b'0' {
            q_pos += 1;
        }
        if q_pos >= queries_buf.len() {
            break;
        }
        let s = parse_u32_fast(&queries_buf, &mut q_pos);
        let t = parse_u32_fast(&queries_buf, &mut q_pos);
        queries.push((s, t));
    }
    let _t_queries = t0.elapsed();

    // 2. Read graph
    let t1 = std::time::Instant::now();
    let graph_path = data_dir.join("graph.txt");
    let graph_buf = match map_file(&graph_path) {
        Ok(b) => b,
        Err(_) => {
            File::create(&output_path).ok();
            return;
        }
    };

    let mut g_pos = 0;
    let n = parse_u32_fast(&graph_buf, &mut g_pos) as usize;
    let m = parse_u32_fast(&graph_buf, &mut g_pos) as usize;
    let t_read_graph = t1.elapsed();

    let num_threads = thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(12);

    let t2 = std::time::Instant::now();
    let out_deg: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let in_deg: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();

    let edges_raw = &graph_buf[g_pos..];
    let total_bytes = edges_raw.len();

    // Determine line-aligned chunk bounds for each thread
    let mut chunk_bounds = Vec::with_capacity(num_threads);
    let chunk_size = total_bytes / num_threads;
    let mut cur_start = 0;
    for t in 0..num_threads {
        let cur_end = if t == num_threads - 1 {
            total_bytes
        } else {
            let target = (t + 1) * chunk_size;
            let mut p = target;
            while p < total_bytes && edges_raw[p] != b'\n' {
                p += 1;
            }
            if p < total_bytes {
                p + 1
            } else {
                total_bytes
            }
        };
        chunk_bounds.push((cur_start, cur_end));
        cur_start = cur_end;
    }

    // Phase 1: Parallel parse edges and count degrees
    let thread_edges: Vec<Vec<(u32, u32, u32)>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(start, end) in &chunk_bounds {
            let slice = &edges_raw[start..end];
            let out_d = &out_deg;
            let in_d = &in_deg;
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity(slice.len() / 18);
                let mut p = 0;
                while p < slice.len() {
                    while p < slice.len() && slice[p] < b'0' {
                        p += 1;
                    }
                    if p >= slice.len() {
                        break;
                    }
                    let u = parse_u32_fast(slice, &mut p);
                    let v = parse_u32_fast(slice, &mut p);
                    let w = parse_u32_fast(slice, &mut p);
                    local.push((u, v, w));
                    out_d[u as usize].fetch_add(1, Ordering::Relaxed);
                    in_d[v as usize].fetch_add(1, Ordering::Relaxed);
                }
                local
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let t_parse = t2.elapsed();

    // Compute CSR offsets
    let t3 = std::time::Instant::now();
    let mut fwd_offsets = vec![0u32; n + 1];
    let mut bwd_offsets = vec![0u32; n + 1];

    let mut sum_out = 0u32;
    for i in 0..n {
        fwd_offsets[i] = sum_out;
        sum_out += out_deg[i].load(Ordering::Relaxed);
    }
    fwd_offsets[n] = sum_out;

    let mut sum_in = 0u32;
    for i in 0..n {
        bwd_offsets[i] = sum_in;
        sum_in += in_deg[i].load(Ordering::Relaxed);
    }
    bwd_offsets[n] = sum_in;

    drop(out_deg);
    drop(in_deg);

    let mut fwd_edges = vec![
        Edge {
            target: 0,
            weight: 0
        };
        m
    ];
    let mut bwd_edges = vec![
        Edge {
            target: 0,
            weight: 0
        };
        m
    ];

    let fwd_pos: Vec<AtomicU32> = fwd_offsets[..n]
        .iter()
        .map(|&off| AtomicU32::new(off))
        .collect();
    let bwd_pos: Vec<AtomicU32> = bwd_offsets[..n]
        .iter()
        .map(|&off| AtomicU32::new(off))
        .collect();

    let fwd_ptr = EdgePtr(fwd_edges.as_mut_ptr());
    let bwd_ptr = EdgePtr(bwd_edges.as_mut_ptr());

    // Phase 2: Parallel CSR placement
    thread::scope(|s| {
        for local in &thread_edges {
            let fp = &fwd_pos;
            let bp = &bwd_pos;
            let f_ptr = fwd_ptr;
            let b_ptr = bwd_ptr;
            s.spawn(move || {
                let f_base = f_ptr.get();
                let b_base = b_ptr.get();
                for &(u, v, w) in local {
                    let slot_f = fp[u as usize].fetch_add(1, Ordering::Relaxed) as usize;
                    unsafe {
                        *f_base.add(slot_f) = Edge {
                            target: v,
                            weight: w,
                        };
                    }
                    let slot_b = bp[v as usize].fetch_add(1, Ordering::Relaxed) as usize;
                    unsafe {
                        *b_base.add(slot_b) = Edge {
                            target: u,
                            weight: w,
                        };
                    }
                }
            });
        }
    });

    drop(thread_edges);
    drop(fwd_pos);
    drop(bwd_pos);

    let graph = Graph {
        n,
        fwd_offsets,
        fwd_edges,
        bwd_offsets,
        bwd_edges,
    };
    let t_csr = t3.elapsed();

    // Phase 3: Parallel query execution (12 threads in L1 cache)
    let t4 = std::time::Instant::now();
    let mut answers = vec![0i64; queries.len()];
    let query_workers = num_threads.min(12);
    let queries_per_thread = (queries.len() + query_workers - 1) / query_workers;

    thread::scope(|s| {
        let g = &graph;
        for (tid, ans_chunk) in answers.chunks_mut(queries_per_thread).enumerate() {
            let q_start = tid * queries_per_thread;
            let q_end = (q_start + ans_chunk.len()).min(queries.len());
            let q_slice = &queries[q_start..q_end];
            s.spawn(move || {
                let mut ctx = WorkerContext::new(g.n);
                for (idx, &(src, tgt)) in q_slice.iter().enumerate() {
                    ans_chunk[idx] = ctx.shortest_path(src, tgt, g);
                }
            });
        }
    });
    let t_queries_exec = t4.elapsed();

    // 4. Output formatting
    let t5 = std::time::Instant::now();
    let mut out_bytes = Vec::with_capacity(answers.len() * 12);
    for &ans in &answers {
        if ans < 0 {
            out_bytes.extend_from_slice(b"-1\n");
        } else if ans == 0 {
            out_bytes.extend_from_slice(b"0\n");
        } else {
            let mut buf = [0u8; 24];
            let mut i = 0;
            let mut v = ans as u64;
            while v > 0 {
                buf[i] = b'0' + (v % 10) as u8;
                v /= 10;
                i += 1;
            }
            while i > 0 {
                i -= 1;
                out_bytes.push(buf[i]);
            }
            out_bytes.push(b'\n');
        }
    }

    std::fs::write(&output_path, out_bytes).expect("failed to write distances.txt");
    let t_write = t5.elapsed();

    eprintln!(
        "TIMING: read_graph={:?} parse={:?} csr={:?} queries_exec={:?} write={:?} TOTAL={:?}",
        t_read_graph,
        t_parse,
        t_csr,
        t_queries_exec,
        t_write,
        t_start.elapsed()
    );
}

const std = @import("std");
const warn = std.debug.warn;

var rng = std.rand.DefaultPrng.init(0x12345678);


pub fn binarySort(comptime T: type, items: []T, lessThan: fn (lhs: T, rhs: T) bool) void {
    {
        if (items.len < 2)
            return;

        var i: usize = 1;

        while (i < items.len) : (i += 1) {
            const v = items[i];
            var l : usize = 0;
            var r : usize = i;
            while (l < r) {
                const m = (l + r) / 2;
                if (lessThan(items[m], v)) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            if (l == i)
                continue;

            var n = i;
            while (n > l) : (n -= 1)
                items[n] = items[n - 1];
            // std.mem.copy(T, items[l+1..i+1], items[l..i]);
            items[l] = v;
        }
    }
}

pub const StackedAllocator = struct {
    pub allocator: std.mem.Allocator,
    child_allocator: *std.mem.Allocator,
    cur_len: usize,
    cur_used: usize,
    cur_alloc: []u8,
    
    pub fn init(child_allocator: *std.mem.Allocator) StackedAllocator {
        return StackedAllocator{
            .allocator = std.mem.Allocator{
                .reallocFn = realloc,
                .shrinkFn = shrink,
            },
            .child_allocator = child_allocator,
            .cur_len = 0,
            .cur_used = 0,
            .cur_alloc = undefined
        };
    }

    pub fn deinit(self: *StackedAllocator) void {
        self.child_allocator.free(self.cur_alloc);
    }

    fn alloc(allocator: *std.mem.Allocator, n: usize, alignment: u29) error{OutOfMemory}![]u8 {
        const self = @fieldParentPtr(StackedAllocator, "allocator", allocator);
        if (self.cur_len == 0) { //nothing was allocated yet
            self.cur_alloc = try self.child_allocator.alloc(u8, n);
            self.cur_len   = self.cur_alloc.len;
            self.cur_used  = self.cur_alloc.len;
            return self.cur_alloc[0..n];
        } else if (self.cur_len < self.cur_used + n) { //we need more memory
            self.cur_alloc = try self.child_allocator.realloc(self.cur_alloc, self.cur_used + n);
            self.cur_len = self.cur_alloc.len;
            const alloc_slice = self.cur_alloc[self.cur_used..self.cur_used + n];
            self.cur_used += n;
            return alloc_slice;
        } else {
            const alloc_slice = self.cur_alloc[self.cur_used..self.cur_used + n];
            self.cur_used += n;
            return alloc_slice;
        }
    }

    fn shrink(allocator: *std.mem.Allocator, old_mem_unaligned: []u8, old_align: u29, new_size: usize, new_align: u29) []u8 {
        const self = @fieldParentPtr(StackedAllocator, "allocator", allocator);
        if ((self.cur_alloc.ptr + self.cur_used) != (old_mem_unaligned.ptr + old_mem_unaligned.len)) {
            warn("Failed ptr align");
            unreachable;
        }

        self.cur_used -= old_mem_unaligned.len;
        return self.cur_alloc[0..0];
    }
    
    fn realloc(allocator: *std.mem.Allocator, old_mem: []u8, old_align: u29, new_size: usize, new_align: u29) ![]u8 {
        return alloc(allocator, new_size, new_align);
    }
};

fn lowerBound(comptime T: type, items: []T, value: T, lessThan: fn(l: T, r: T) bool) usize {
    if (items.len < 2)
        return 0;

    var l : usize = 0;
    var r : usize = items.len;
    while (l + 1 < r) {
        const m = (l + r) / 2;
        if (lessThan(items[m], value)) {//we want l to be _past_ the value (for sorting purpose)
            l = m;
        } else {
            r = m;
        }
    }
    return l;
}

fn upperBound(comptime T: type, items: []T, value: T, lessThan: fn(l: T, r: T) bool) usize {
    std.debug.assert(items.len != 0);
    if (items.len < 2)
        return 0;

    var l : usize = 0;
    var r : usize = items.len;
    while (l < r) {
        const m = (l + r) / 2;
        if (lessThan(items[m], value)) {//we want l to be _past_ the value (for sorting purpose)
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

//mergeSort assumes items1 follows right before items2 in memory
//merge sort allocates to can return an error
fn mergeSortLeft(comptime T: type, items1: []T, items2: []T, lessThan: fn(l: T, r: T) bool, allocator : *std.mem.Allocator) ![]T {
    const tmp = try allocator.alloc(T, items1.len);

    defer allocator.free(tmp);
    std.mem.copy(T, tmp, items1);


    // const startSrc1 = binarySearch(T, items1, items2[0], lessThan);

    // warn("{} {} {} {}\n", items1.len, startSrc1, items2[0], items1[0]);

    var src1Ptr = tmp.ptr;
    const src1End = tmp.ptr + tmp.len;
    
    var src2Ptr = items2.ptr;
    const src2End = items2.ptr + items2.len;

    var targetPtr = items1.ptr;

    while (true) {
        if (lessThan(src1Ptr[0], src2Ptr[0])) {
            targetPtr[0] = src1Ptr[0];
            src1Ptr += 1;
        } else {
            targetPtr[0] = src2Ptr[0];
            src2Ptr += 1;
        }

        targetPtr += 1;

        if (src1Ptr == src1End) {//no need to copy, it's already there
            break;
        } 
        if (src2Ptr == src2End) { //copy rest of the tmp data to the end
            const leftBytes = @ptrToInt(src1End) - @ptrToInt(src1Ptr);
            @memcpy(@ptrCast([*]u8, targetPtr), @ptrCast([*]u8, src1Ptr), leftBytes);
            break; 
        }

    }
    return items1.ptr[0..items1.len + items2.len];
}


fn mergeSortRight(comptime T: type, items1: []T, items2: []T, lessThan: fn(l: T, r: T) bool, allocator : *std.mem.Allocator) ![]T {
    const tmp = try allocator.alloc(T, items2.len);
    defer allocator.free(tmp);

    std.mem.copy(T, tmp, items2);


    // const startSrc1 = binarySearch(T, items1, items2[0], lessThan);

    //We will walk backwards
    var src1Ptr = items1.ptr + items1.len - 1;
    const src1Start = items1.ptr;
    
    var src2Ptr = tmp.ptr + tmp.len - 1;
    const src2Start = tmp.ptr;

    var targetPtr = items2.ptr + tmp.len - 1;

    while (true) {
        if (!lessThan(src2Ptr[0], src1Ptr[0])) {
            targetPtr[0] = src2Ptr[0];
            src2Ptr -= 1;
        } else {
            targetPtr[0] = src1Ptr[0];
            src1Ptr -= 1;
        }

        targetPtr -= 1;

        if (src2Ptr + 1 == src2Start) {//no need to copy, it's already there
            break;
        } 
        if (src1Ptr + 1 == src1Start) { //copy rest of the tmp data to the end
            const leftBytes = @ptrToInt(src2Ptr + 1) - @ptrToInt(src2Start);
            @memcpy(@ptrCast([*]u8, src1Start), @ptrCast([*]u8, src2Start), leftBytes);
            break; 
        }

    }
    return items1.ptr[0..items1.len + items2.len];
}

fn mergeSort(comptime T: type, items1: []T, items2: []T, lessThan: fn(l: T, r: T) bool, allocator : *std.mem.Allocator) ![]T {
    if (items1.len == 0)
        return items2;
    
    if (items2.len == 0)
        return items1;

    const full_array = items1.ptr[0..items1.len + items2.len];
    
    const start_items1  = 0;//lowerBound(T, items1, items2[0], lessThan);
    const end_items2    = items2.len;//upperBound(T, items2, items1[items1.len - 1], lessThan);

    const sort_items1 = items1[start_items1..];
    const sort_items2 = items2[0..end_items2];
    if (sort_items1.len == 0 or
        sort_items2.len == 0)
        return full_array;

    if (sort_items1.len < sort_items2.len) {
        _ = try mergeSortLeft(T, sort_items1, sort_items2, lessThan, allocator);
    } else {
        _ = try mergeSortRight(T, sort_items1, sort_items2, lessThan, allocator);
    }
    return full_array;
}


fn timSortNextRun(comptime T: type, items: []T, lessThan: fn(l: T, r: T) bool, min_run : usize) []T {
    if (items.len < 2)
        return items;
    
    var cur = items.ptr;
    var next = items.ptr + 1;

    var len : usize = 1;
    const seq_len = items.len;
    
    //find natural run
    if (lessThan(cur[0], next[0])) { //non-descending
        while (true) {
            cur += 1;
            next += 1;
            len += 1;
            if (! (len < seq_len and lessThan(cur[0], next[0])))
                break;
        }
    } else { //descending
        while (true) {
            cur += 1;
            next += 1;
            len += 1;
            if (! (len < seq_len and lessThan(next[0], cur[0])))
                break;
        }
        std.mem.reverse(T, items[0..len]);
    }

    //extent run if needed, and insertion sort it
    if (len < min_run and len < seq_len) {
        const extent = std.math.min(min_run, seq_len);
        // std.sort.insertionSort(T, items[0..extent], lessThan);
        binarySort(T, items[0..extent], lessThan);
        return items[0..extent];
    }

    //return run slice
    return items[0..len];
}

fn calculateMinRun(len: usize) usize {
    var r : usize = 0;
    var n = len;
    while (n >= 32) {
        r |= n & 1;
        n >>= 1;
    }
    return r + n;
}

fn timSort(comptime T: type, items: []T, lessThan: fn(l: T, r: T) bool) !void {
    var allocator = std.heap.direct_allocator;

    
    const min_run = calculateMinRun(items.len);

    //we know the max size of the Run stack, since every run is at least min_run in size, and we know that number of items
    //we estimate the size and double it since ArrayList might use a doubling allocation approach
    const stack_size = 2 * @sizeOf([]T) * (items.len / min_run + 1);

    // allocate the buffer for the run stack 
    var stack_buf = try allocator.alloc(u8, stack_size);
    defer allocator.free(stack_buf);

    var run_stack_alloc = &std.heap.FixedBufferAllocator.init(stack_buf).allocator;
    var stack = std.ArrayList([]T).init(run_stack_alloc);
    // var stack = std.ArrayList([]T).init(&allocator);

    // To perform the mergeSort, we need to make allocations to temporarily store the values
    // Instead of using slow direct allocators, we use a stack allocator that keeps memory mapped
    // And grows it only if needed, to save on kernel calls

    var sa = StackedAllocator.init(allocator);
    defer sa.deinit();
    var merging_allocator = &sa.allocator;

    const first_run = timSortNextRun(T, items, lessThan, min_run);
    try stack.append(first_run);
    var whats_left = items[first_run.len..];

    while (true) {
        if (whats_left.len == 0)
            break;
        if (stack.len < 3) {
            const next_run = timSortNextRun(T, whats_left, lessThan, min_run);
            try stack.append(next_run);
            whats_left = whats_left[next_run.len..];
            continue;
        }

        if (stack.len >= 3) {
            const X = stack.at(stack.len - 1); //most recent in stack, comes after r2
            const Y = stack.at(stack.len - 2);
            const Z = stack.at(stack.len - 3);


            if (Z.len <= Y.len + X.len) {
                const new_slice = try mergeSort(T, Z, Y, lessThan, merging_allocator);
                _ = stack.pop();
                _ = stack.pop();
                stack.set(stack.len - 1, new_slice);
                try stack.append(X);
            } else if (Y.len <= X.len) {
                const new_slice = try mergeSort(T, Y, X, lessThan, merging_allocator);
                _ = stack.pop();
                stack.set(stack.len - 1, new_slice);
            } else {
                const next_run = timSortNextRun(T, whats_left, lessThan, min_run);
                try stack.append(next_run);
                whats_left = whats_left[next_run.len..];
                continue;
            }
        }
    }

    while (stack.len > 1) {
        const r1 = stack.at(stack.len - 1);
        const r2 = stack.at(stack.len - 2);
        
        stack.set(stack.len - 2, try mergeSort(T, r2, r1, lessThan, merging_allocator));
        _ = stack.pop();
    }
}

test "Test Sequential" {
    var allocator = std.heap.direct_allocator;
    
    const len = 20;
    var values = try allocator.alloc(f64, len);
    const seq_1 = values[0..15];
    const seq_2 = values[15..];
    
    for (seq_1) |*v, n| {
        v.* = @intToFloat(f64, n);
    }
    for (seq_2) |*v, n| {
        v.* = @intToFloat(f64, n);
    }

    const sorted_values = mergeSortLeftInPlace(f64, seq_1, seq_2, std.sort.asc(f64));
    
    // for (sorted_values) |v| {
    //     warn("{}\n", v);
    // }
}

pub fn main() anyerror!void {
    const N = 40000000;
    std.debug.warn("allocating\n");

    var allocator = std.heap.direct_allocator;
    // const f64 = f64;

    var values = try allocator.alloc(f64, N);
    var values2 = try allocator.alloc(f64, N);

    for (values) |_, n| {
        const val = @intToFloat(f64, rng.random.range(i64, -1000, 1000)); //fully random
        // const val = @intToFloat(f64, @intCast(i64, n) + rng.random.range(i64, -10, 10)); //slightly random, mostly ascending
        // const val = @intToFloat(f64, -@intCast(i64, n) + rng.random.range(i64, -10, 10)); //slightly random, mostly descending
        // warn("{} ", val);
        values[n] = val;
        values2[n] = val;
    }

    //Run default sort
    std.debug.warn("Running default sort\n");
    const start_time_default = std.time.milliTimestamp();
    std.sort.sort(f64, values, std.sort.asc(f64));
    warn("Default sort took: {}\n", std.time.milliTimestamp() - start_time_default);

    //Run TimSort
    std.debug.warn("Running TimSort sort\n");
    const start_time_tim = std.time.milliTimestamp();
    try timSort(f64, values2, std.sort.asc(f64));
    warn("Tim sort took: {}\n", std.time.milliTimestamp() - start_time_tim);

    //Check if values correspond
    for (values) |_, n| {
        // warn("{} {}\n", values[n], values2[n]);
        std.debug.assert(values[n] == values2[n]);
    }
}

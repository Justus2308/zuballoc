# zuballoc
A hard realtime O(1) allocator for sub-allocating memory regions with minimal fragmentation.

This implementation is based on the [OffsetAllocator by sebbbi](https://github.com/sebbbi/OffsetAllocator).

Uses 256 bins following the 8-bit floating point distribution (3 bits mantissa, 5 bits exponent).

The allocation metadata can optionally be stored outside of the sub-allocated memory region, making this allocator suitable for sub-allocating any resources, such as GPU heaps, buffers and arrays.

The allocator is fully compatible with the Zig `Allocator` interface and provides a `.allocator()` function.

## Implementation

The goal of this allocator is to provide consistent and predictable allocation speeds without any outliers to make it suitable for realtime applications with tight time constraints.

This is achieved by using a two-stage bit set and bitwise operations (mainly count-trailing-zeroes) instead of loops to find free nodes that contain enough memory to fulfill an allocation request.

The general architecture and implementation is based on the aforementioned [OffsetAllocator](https://github.com/sebbbi/OffsetAllocator) which is itself based on the [Two-Level Segregated Fit (TLSF) algorithm](https://www.researchgate.net/publication/4080369_TLSF_A_new_dynamic_memory_allocator_for_real-time_systems).

TLSF was first developed in the early 2000s as a successor to the (much older) buddy allocator and its derivitives which often tend to produce a lot of memory fragmentation. [It has been shown](https://www.researchgate.net/publication/234785757_A_comparison_of_memory_allocators_for_real-time_applications) to provide the best balance between response time and fragmentation compared to other realtime allocation schemes.

When the allocator recieves an allocation request, it will first calculate the appropriate bin for the given size. This is done by first converting the requested size to an 8-bit float (3-bit mantissa, 5-bit exponent) and then reinterpreting the result as an integer. This method results in 256 total bins whose size follows the (logarithmic) floating point distribution. The first 17 bins are exact bins which is very nice for efficiently handling small allocations, especially compared to other binning methods like power-of-two bins.

Then it tries to pop a node from the free list of the calculated bin. If there is none, it will instead query the bitsets for the next lowest set bit, which indicates that there are free nodes available in that bin. They will be oversized, but they will fit the allocation.

If there is any excess memory (because of an oversized bin or alignment) it will put back into the free list as a new node at the appropriate bin to combat internal fragmentation.

The metadata for each node keeps track of its direct in-memory neighbors and will try to coalesce itself with them when it gets freed.

All of this works without any loops which results in guaranteed O(1) time complexity.

## Usage

### Add this project to yours as a dependency:

1. Run this command:

```sh
zig fetch --save git+https://github.com/Justus2308/zuballoc.git
```

2. Import the module into your `build.zig` file

```zig
const zuballoc_dependency = b.dependency("zuballoc", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zuballoc", zuballoc_dependency.module("zuballoc"));
```

### Use as a module in your code:

```zig
const std = @import("std");
const zuballoc = @import("zuballoc");

pub fn main() !void {
    var buffer: [1024]u8 = undefined;
    var sub_allocator = try zuballoc.SubAllocator.init(std.heap.smp_allocator, &buffer, 256);
    defer sub_allocator.deinit(std.heap.smp_allocator);

    // Allocations with external metadata
    const slice_allocation = try sub_allocator.allocWithMetadata(u8, 116);
    defer sub_allocator.freeWithMetadata(slice_allocation);
    const slice: []u8 = slice_allocation.get();

    const single_allocation = try sub_allocator.createWithMetadata(u64);
    defer sub_allocator.destroyWithMetadata(single_allocation);
    const item: *u64 = single_allocation.get();

    // Allocations with intrusive metadata
    const allocator = sub_allocator.allocator();

    const memory = try allocator.alloc(u16, 12);
    defer allocator.free(memory);
    
    var list = std.ArrayListUnmanaged(u32).empty;
    defer list.deinit(allocator);
    
    try list.append(allocator, 123);
    
    _ = .{ slice, item };
}
```

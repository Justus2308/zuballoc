# zuballoc
A hard realtime O(1) allocator for sub-allocating memory regions with minimal fragmentation.

This implementation is based on the [OffsetAllocator by sebbbi](https://github.com/sebbbi/OffsetAllocator).

Uses 256 bins following the 8-bit floating point distribution (3 bits mantissa, 5 bits exponent).

The allocation metadata can optionally be stored outside of the sub-allocated memory region, making this allocator suitable for sub-allocating any resources, such as GPU heaps, buffers and arrays.

The allocator is fully compatible with the Zig `Allocator` interface and provides a `.allocator()` function.

## Usage

##### Add this project to yours as a dependency:

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

##### Use as a module in your code:

```zig
const std = @import("std");
const zuballoc = @import("zuballoc");

pub fn main() !void {
    var buffer: [1024]u8 = undefined;
    var sub_allocator = try zuballoc.SubAllocator.init(std.heap.smp_allocator, &buffer, 256);
    defer sub_allocator.deinit();

    // Allocations with external metadata
    const slice_allocation = try sub_allocator.allocWithMetadata(u8, 116);
    defer sub_allocator.freeWithMetadata(slice_allocation);
    const slice: []u8 = slice_allocation.slice();

    const single_allocation = try sub_allocator.createWithMetadata(u64);
    defer sub_allocator.destroyWithMetadata(single_allocation);
    const item: *u64 = single_allocation.ptr;

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
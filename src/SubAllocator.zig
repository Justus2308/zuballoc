ptr: [*]u8,
size: Size,
free_storage: Size,

used_bins_top: std.bit_set.IntegerBitSet(top_bin_count),
used_bins: [top_bin_count]std.bit_set.IntegerBitSet(bins_per_leaf),
bin_indices: [leaf_bins_count]Node.Index,

nodes: Nodes,
is_used: std.DynamicBitSetUnmanaged,
free_nodes: Indices,
free_offset: Node.Index,

const top_bin_count = (1 << f8.exponent_bit_count);
const bins_per_leaf = (1 << f8.mantissa_bit_count);
const leaf_bins_count = (top_bin_count * bins_per_leaf);

const Self = @This();

const Size = u32;
const Log2Size = math.Log2Int(Size);

const Node = struct {
    data_offset: Size,
    data_size: Size,
    bin_list_prev: Index,
    bin_list_next: Index,
    neighbor_prev: Index,
    neighbor_next: Index,

    pub const IndexType = u32;

    pub const Index = enum(IndexType) {
        unused = math.maxInt(IndexType),
        _,

        pub inline fn from(val: IndexType) Index {
            const index: Index = @enumFromInt(val);
            assert(index != .unused);
            return index;
        }
        pub inline fn asInt(index: Index) IndexType {
            assert(index != .unused);
            return @intFromEnum(index);
        }

        /// Allows wrapping from `.unused` to 0, but not incrementing to `.unused`.
        pub inline fn incr(index: *Index) void {
            const n = (@intFromEnum(index.*) +% 1);
            index.* = @enumFromInt(n);
            assert(index.* != .unused);
        }
        /// Allows wrapping to `.unused`, but not decrementing from `.unused`.
        pub inline fn decr(index: *Index) void {
            assert(index.* != .unused);
            const n = (@intFromEnum(index.*) -% 1);
            index.* = @enumFromInt(n);
        }
    };
    pub const Log2Index = math.Log2Int(IndexType);
};

const is_debug = (builtin.mode == .Debug);
const is_safe = (is_debug or builtin.mode == .ReleaseSafe);

// Make automatic OOB checks possible in safe builds
const Nodes = if (is_safe) []Node else [*]Node;
const Indices = if (is_safe) []Node.Index else [*]Node.Index;

const f8 = packed struct(u8) {
    mantissa: u3,
    exponent: u5,

    pub const Mantissa = u3;
    pub const Exponent = u5;

    /// 0b111
    pub const max_mantissa = math.maxInt(Mantissa);
    /// 0b11111
    pub const max_exponent = math.maxInt(Exponent);

    pub const mantissa_bit_count = @as(comptime_int, @typeInfo(Mantissa).int.bits);
    pub const exponent_bit_count = @as(comptime_int, @typeInfo(Exponent).int.bits);

    pub inline fn asInt(float: f8) u8 {
        return @bitCast(float);
    }
};

const FloatFromSizeMode = enum { floor, ceil };
fn floatFromSize(comptime mode: FloatFromSizeMode, size: Size) f8 {
    var float = @as(f8, @bitCast(@as(u8, 0)));

    if (size <= f8.max_mantissa) {
        float.mantissa = @truncate(size);
    } else {
        const leading_zeroes: Log2Size = @truncate(@clz(size));
        const highest_set_bit = (math.maxInt(Log2Size) - leading_zeroes);

        const mantissa_start_bit = highest_set_bit - f8.mantissa_bit_count;
        float.exponent = @intCast(mantissa_start_bit + 1);
        float.mantissa = @truncate(size >> mantissa_start_bit);

        const low_bit_mask = (@as(Size, 1) << mantissa_start_bit) - 1;

        // Round up
        if (mode == .ceil and (size & low_bit_mask) != 0) {
            float.mantissa, const carry = @addWithOverflow(float.mantissa, 1);
            float.exponent += carry;
        }
    }

    return float;
}

fn sizeFromFloat(float: f8) Size {
    if (float.exponent == 0) {
        return float.mantissa;
    } else {
        return (@as(Size, float.mantissa) | (@as(Size, 1) << f8.mantissa_bit_count)) << (float.exponent - 1);
    }
}

fn findFirstSetAfter(mask: anytype, start_index: math.Log2Int(@TypeOf(mask))) ?math.Log2Int(@TypeOf(mask)) {
    const mask_before_start_index = (@as(@TypeOf(mask), 1) << start_index) - 1;
    const mask_after_start_index = ~mask_before_start_index;
    const bits_after = (mask & mask_after_start_index);
    if (bits_after == 0) {
        return null;
    }
    return @truncate(@ctz(bits_after));
}

pub fn init(gpa: Allocator, buffer: []u8, max_alloc_count: Node.IndexType) Allocator.Error!Self {
    assert(max_alloc_count > 0);
    var self: Self = undefined;
    self.ptr = buffer.ptr;
    self.size = @intCast(buffer.len);
    const node_count = (max_alloc_count + 1);
    self.nodes = @ptrCast(try gpa.alloc(Node, node_count));
    errdefer gpa.free(self.nodes[0..node_count]);
    self.free_nodes = @ptrCast(try gpa.alloc(Node.Index, node_count));
    errdefer gpa.free(self.free_nodes[0..node_count]);
    self.is_used = try .initEmpty(gpa, node_count);
    errdefer self.is_used.deinit(gpa);
    self.reset();
    return self;
}

pub fn deinit(self: *Self, gpa: Allocator) void {
    const node_count = (self.maxAllocCount() + 1);
    gpa.free(self.nodes[0..node_count]);
    gpa.free(self.free_nodes[0..node_count]);
    self.is_used.deinit(gpa);
}

pub fn reset(self: *Self) void {
    self.free_storage = 0;
    self.used_bins_top = .initEmpty();
    self.free_offset = .from(@intCast(self.maxAllocCount()));

    @memset(&self.used_bins, .initEmpty());
    @memset(&self.bin_indices, .unused);

    const node_count = (self.maxAllocCount() + 1);

    // Freelist is a stack. Nodes in inverse order so that [0] pops first.
    for (0..node_count) |i| {
        self.free_nodes[i] = .from(@intCast(self.maxAllocCount() - @as(u32, @intCast(i))));
    }
    @memset(self.nodes[0..node_count], Node{
        .data_offset = 0,
        .data_size = 0,
        .bin_list_prev = .unused,
        .bin_list_next = .unused,
        .neighbor_prev = .unused,
        .neighbor_next = .unused,
    });
    self.is_used.unsetAll();

    // Start state: Whole storage as one big node
    // Algorithm will split remainders and push them back as smaller nodes
    _ = self.insertNodeIntoBin(self.size, 0);
}

pub fn ownsPtr(self: Self, ptr: [*]const u8) bool {
    return sliceContainsPtr(self.ptr[0..self.size], ptr);
}

pub fn ownsSlice(self: Self, slice: []const u8) bool {
    return sliceContainsSlice(self.ptr[0..self.size], slice);
}

pub fn maxAllocCount(self: Self) u32 {
    return @intCast(self.is_used.bit_length - 1);
}

pub fn totalFreeSpace(self: Self) Size {
    if (self.free_offset == .unused) {
        return 0;
    }
    return self.free_storage;
}

pub fn largestFreeRegion(self: Self) Size {
    if (self.free_offset == .unused or self.used_bins_top.count() == 0) {
        return 0;
    }
    const top_bin_index = self.used_bins_top.findLastSet() orelse 0;
    const leaf_bin_index = self.used_bins[top_bin_index].findLastSet() orelse 0;
    const float = f8{ .mantissa = leaf_bin_index, .exponent = top_bin_index };
    const largest_free_region = sizeFromFloat(float);
    assert(self.free_storage >= largest_free_region);
    return largest_free_region;
}

pub const StorageReport = struct {
    free_regions: [leaf_bins_count]Region,

    pub const Region = struct {
        size: Size,
        count: u32,
    };

    pub fn print(report: StorageReport) void {
        std.debug.print("===== STORAGE REPORT =====\n", .{});
        for (report.free_regions) |free_region| {
            if (free_region.count > 0) {
                std.debug.print("size={d},count={d}\n", .{
                    free_region.size,
                    free_region.count,
                });
            }
        }
        std.debug.print("==========================\n", .{});
    }
};

pub fn storageReport(self: Self) StorageReport {
    var report: StorageReport = .{ .free_regions = undefined };
    for (0..leaf_bins_count) |i| {
        var count: u32 = 0;
        var node_index = self.bin_indices[i];
        while (node_index != .unused) : (node_index = self.nodes[node_index.asInt()].bin_list_next) {
            count += 1;
        }
        report.free_regions[i] = .{ .size = sizeFromFloat(@bitCast(@as(u8, @intCast(i)))), .count = count };
    }
    return report;
}

fn insertNodeIntoBin(self: *Self, size: Size, data_offset: Size) Node.Index {
    // Round down to bin index to ensure that bin >= alloc
    const bin_index = floatFromSize(.floor, size);

    const top_bin_index = bin_index.exponent;
    const leaf_bin_index = bin_index.mantissa;

    // Bin was empty before?
    if (self.bin_indices[bin_index.asInt()] == .unused) {
        // Set bin mask bits
        self.used_bins[top_bin_index].set(leaf_bin_index);
        self.used_bins_top.set(top_bin_index);
    }

    // Take a freelist node and insert on top of the bin linked list (next = old top)
    const top_node_index = self.bin_indices[bin_index.asInt()];
    const node_index = self.free_nodes[self.free_offset.asInt()];
    log.debug("Getting node {d} from freelist[{d}] (insertNodeIntoBin)", .{ node_index.asInt(), self.free_offset.asInt() });
    self.free_offset.decr();

    self.nodes[node_index.asInt()] = .{
        .data_offset = data_offset,
        .data_size = size,
        .bin_list_prev = .unused,
        .bin_list_next = top_node_index,
        .neighbor_prev = .unused,
        .neighbor_next = .unused,
    };
    if (top_node_index != .unused) {
        self.nodes[top_node_index.asInt()].bin_list_prev = node_index;
    }
    self.bin_indices[bin_index.asInt()] = node_index;

    self.free_storage += size;

    log.debug("Free storage: {d} (+{d}) (insertNodeIntoBin)", .{ self.free_storage, size });

    return node_index;
}

fn removeNodeFromBin(self: *Self, node_index: Node.Index) void {
    const node: *Node = &self.nodes[node_index.asInt()];

    if (node.bin_list_prev != .unused) {
        // Easy case: We have previous node. Just remove this node from the middle of the list.
        self.nodes[node.bin_list_prev.asInt()].bin_list_next = node.bin_list_next;
        if (node.bin_list_next != .unused) {
            self.nodes[node.bin_list_next.asInt()].bin_list_prev = node.bin_list_prev;
        }
    } else {
        // Hard case: We are the first node in a bin. Find the bin.

        // Round down to bin index to ensure that bin >= alloc
        const bin_index = floatFromSize(.floor, node.data_size);

        const top_bin_index = bin_index.exponent;
        const leaf_bin_index = bin_index.mantissa;

        self.bin_indices[bin_index.asInt()] = node.bin_list_next;
        if (node.bin_list_next != .unused) {
            self.nodes[node.bin_list_next.asInt()].bin_list_prev = .unused;
        }

        // Bin empty?
        if (self.bin_indices[bin_index.asInt()] == .unused) {
            // Remove a leaf bin mask bit
            self.used_bins[top_bin_index].unset(leaf_bin_index);

            // All leaf bins empty?
            if (self.used_bins[top_bin_index].mask == 0) {
                // Remove a top bin mask bit
                self.used_bins_top.unset(top_bin_index);
            }
        }

        // Insert the node to freelist
        self.free_offset.incr();
        log.debug("Putting node {d} into freelist[{d}] (removeNodeFromBin)", .{ node_index.asInt(), self.free_offset.asInt() });
        self.free_nodes[self.free_offset.asInt()] = node_index;

        self.free_storage -= node.data_size;

        log.debug("Free storage: {d} (-{d}) (removeNodeFromBin)", .{ self.free_storage, node.data_size });
    }
}

const AllocationKind = union(enum) {
    pointer,
    slice: ?Alignment,
};
fn Allocation(comptime T: type, comptime kind: AllocationKind) type {
    return switch (kind) {
        .pointer => struct {
            ptr: *T,
            metadata: Node.Index,
        },
        .slice => |alignment| slice: {
            const alignment_in_bytes = if (alignment) |a| a.toByteUnits() else @alignOf(T);
            break :slice struct {
                ptr: [*]align(alignment_in_bytes) T,
                len: Size,
                metadata: Node.Index,

                pub inline fn slice(self: @This()) []align(alignment_in_bytes) T {
                    return self.ptr[0..self.len];
                }
            };
        },
    };
}

pub fn createWithMetadata(self: *Self, comptime T: type) Allocator.Error!Allocation(T, .pointer) {
    const size: Size = @sizeOf(T);
    const inner_allocation = self.innerAlloc(.external, size, @alignOf(T));
    if (inner_allocation.isOutOfMemory()) {
        return Allocator.Error.OutOfMemory;
    }
    const ptr: *T = @ptrCast(@alignCast(self.ptr[inner_allocation.offset..][0..@sizeOf(T)]));
    return .{
        .ptr = ptr,
        .metadata = inner_allocation.metadata,
    };
}

pub fn destroyWithMetadata(self: *Self, allocation: anytype) void {
    const ptr_info = @typeInfo(@FieldType(@TypeOf(allocation), "ptr")).pointer;
    if (ptr_info.size != .one) @compileError("allocation must be of type pointer");

    assert(@inComptime() or self.ownsSlice(mem.asBytes(allocation.ptr)));
    self.innerFree(allocation.metadata);
}

pub fn allocWithMetadata(self: *Self, comptime T: type, n: Size) Allocator.Error!Allocation(T, .{ .slice = null }) {
    return self.alignedAllocWithMetadata(T, null, n);
}

pub fn alignedAllocWithMetadata(
    self: *Self,
    comptime T: type,
    comptime alignment: ?Alignment,
    n: Size,
) Allocator.Error!Allocation(T, .{ .slice = alignment }) {
    const alignment_in_bytes: Size = comptime @intCast(if (alignment) |a| a.toByteUnits() else @alignOf(T));
    const size: Size = math.cast(Size, (n * @sizeOf(T))) orelse return Allocator.Error.OutOfMemory;
    const inner_allocation = self.innerAlloc(.external, size, alignment_in_bytes);
    if (inner_allocation.isOutOfMemory()) {
        return Allocator.Error.OutOfMemory;
    }
    const ptr: [*]align(alignment_in_bytes) T = @ptrCast(@alignCast(self.ptr[inner_allocation.offset..]));
    return .{
        .ptr = ptr,
        .len = n,
        .metadata = inner_allocation.metadata,
    };
}

pub fn resizeWithMetadata(self: *Self, allocation_ptr: anytype, new_len: Size) bool {
    const ptr_info = @typeInfo(@FieldType(std.meta.Child(@TypeOf(allocation_ptr)), "ptr")).pointer;
    if (ptr_info.size != .many) @compileError("allocation must be of type slice");

    assert(@inComptime() or self.ownsSlice(mem.sliceAsBytes(allocation_ptr.ptr[0..allocation_ptr.len])));

    const old_size = (allocation_ptr.len * @sizeOf(ptr_info.child));
    const new_size = (new_len * @sizeOf(ptr_info.child));
    const ok = self.innerResize(allocation_ptr.metadata, old_size, new_size);
    if (ok) {
        allocation_ptr.len = new_len;
    }
    return ok;
}

pub fn freeWithMetadata(self: *Self, allocation: anytype) void {
    const ptr_info = @typeInfo(@FieldType(@TypeOf(allocation), "ptr")).pointer;
    if (ptr_info.size != .many) @compileError("allocation must be of type slice");

    assert(@inComptime() or self.ownsSlice(mem.sliceAsBytes(allocation.ptr[0..allocation.len])));
    self.innerFree(allocation.metadata);
}

const InnerAllocation = struct {
    offset: Size,
    metadata: Metadata,

    pub const oom = InnerAllocation{
        .offset = undefined,
        .metadata = .unused,
    };

    pub const Metadata = Node.Index;

    pub inline fn isOutOfMemory(inner_allocation: InnerAllocation) bool {
        return (inner_allocation.metadata == .unused);
    }
};

const MetadataKind = enum { external, embedded };

fn innerAlloc(self: *Self, comptime metadata_kind: MetadataKind, size: Size, alignment: Size) InnerAllocation {
    if (self.free_offset == .unused) {
        @branchHint(.unlikely);
        return .oom;
    }

    const effective_alignment = switch (metadata_kind) {
        .external => alignment,
        .embedded => @max(alignment, @alignOf(InnerAllocation.Metadata)),
    };
    const effective_size = switch (metadata_kind) {
        .external => size,
        .embedded => (effective_alignment + size),
    };
    const size_alignable = (effective_size + effective_alignment - 1);

    // Round up to bin index to ensure that alloc >= bin
    // Gives us min bin index that fits the size
    const min_bin_index = floatFromSize(.ceil, size_alignable);

    const min_top_bin_index = min_bin_index.exponent;
    const min_leaf_bin_index = min_bin_index.mantissa;

    const top_bin_index: f8.Exponent, const leaf_bin_index: f8.Mantissa = indices: {
        // If top bin exists, scan its leaf bin. This can fail.
        if (self.used_bins_top.isSet(min_top_bin_index)) {
            if (findFirstSetAfter(self.used_bins[min_top_bin_index].mask, min_leaf_bin_index)) |leaf_bin_index| {
                break :indices .{ min_top_bin_index, leaf_bin_index };
            }
        }

        // If we didn't find space in top bin, we search top bin from +1
        const top_bin_index = findFirstSetAfter(self.used_bins_top.mask, (min_top_bin_index + 1)) orelse {
            // OOM
            return .oom;
        };

        // All leaf bins here fit the alloc, since the top bin was rounded up. Start leaf search from bit 0.
        // NOTE: This search can't fail since at least one leaf bit was set because the top bit was set.
        const leaf_bin_index: f8.Mantissa = @intCast(self.used_bins[top_bin_index].findFirstSet().?);

        break :indices .{ top_bin_index, leaf_bin_index };
    };

    const bin_index = f8.asInt(.{
        .mantissa = leaf_bin_index,
        .exponent = top_bin_index,
    });

    // Pop the top node of the bin. Bin top = node.next.
    // We also need to account for alignment by offsetting
    // into the actual allocation until we are aligned.
    const node_index = self.bin_indices[bin_index];
    const node: *Node = &self.nodes[node_index.asInt()];
    const node_total_size = node.data_size;

    const alignment_padding = padding: {
        const base_addr = (@intFromPtr(self.ptr) + node.data_offset);
        const aligned_addr = mem.alignForward(usize, base_addr, effective_alignment);
        const padding: Size = @intCast(aligned_addr - base_addr);
        if (metadata_kind == .embedded and padding < @sizeOf(InnerAllocation.Metadata)) {
            // Metadata does not fit into alignment padding yet
            break :padding (padding + effective_alignment);
        } else {
            break :padding padding;
        }
    };
    const size_aligned = (alignment_padding + size);
    assert(size_aligned <= node_total_size);

    self.is_used.set(node_index.asInt());
    self.bin_indices[bin_index] = node.bin_list_next;
    if (node.bin_list_next != .unused) {
        self.nodes[node.bin_list_next.asInt()].bin_list_prev = .unused;
    }
    self.free_storage -= node_total_size;

    log.debug("Free storage: {d} (-{d}) (alloc)", .{ self.free_storage, node_total_size });

    // Bin empty?
    if (self.bin_indices[bin_index] == .unused) {
        // Remove a leaf bin mask bit
        self.used_bins[top_bin_index].unset(leaf_bin_index);

        // All leaf bins empty?
        if (self.used_bins[top_bin_index].mask == 0) {
            // Remove a top bin mask bit
            self.used_bins_top.unset(top_bin_index);
        }
    }

    const offset = (node.data_offset + alignment_padding);

    // Push back remainder N elements to a lower bin
    const remainder_size_rhs = (node_total_size - size_aligned);
    if (remainder_size_rhs > 0) {
        const new_node_index = self.insertNodeIntoBin(remainder_size_rhs, (node.data_offset + size_aligned));
        node.data_size -= remainder_size_rhs;

        // Link nodes next to each other so that we can merge them later if both are free
        // And update the old next neighbor to point to the new node (in middle)
        if (node.neighbor_next != .unused) {
            self.nodes[node.neighbor_next.asInt()].neighbor_prev = new_node_index;
            self.nodes[new_node_index.asInt()].neighbor_next = node.neighbor_next;
        }
        self.nodes[new_node_index.asInt()].neighbor_prev = node_index;
        node.neighbor_next = new_node_index;
    }

    const remainder_size_lhs = switch (metadata_kind) {
        .external => alignment_padding,
        .embedded => (alignment_padding - @sizeOf(InnerAllocation.Metadata)),
    };
    if (remainder_size_lhs > 0) {
        const new_node_index = self.insertNodeIntoBin(remainder_size_lhs, node.data_offset);
        node.data_offset += remainder_size_lhs;
        node.data_size -= remainder_size_lhs;

        // Link nodes next to each other so that we can merge them later if both are free
        // And update the old next neighbor to point to the new node (in middle)
        if (node.neighbor_prev != .unused) {
            self.nodes[node.neighbor_prev.asInt()].neighbor_next = new_node_index;
            self.nodes[new_node_index.asInt()].neighbor_prev = node.neighbor_prev;
        }
        self.nodes[new_node_index.asInt()].neighbor_next = node_index;
        node.neighbor_prev = new_node_index;
    }

    return InnerAllocation{
        .offset = offset,
        .metadata = node_index,
    };
}

fn innerResize(self: *Self, metadata: InnerAllocation.Metadata, old_size: Size, new_size: Size) bool {
    const node_index = metadata;

    assert(self.is_used.isSet(node_index.asInt()));
    const node: *Node = &self.nodes[node_index.asInt()];

    const size_diff = (@as(i64, new_size) - @as(i64, old_size));

    if (size_diff > 0) {
        if (node.neighbor_next == .unused or self.is_used.isSet(node.neighbor_next.asInt())) {
            return false;
        }
        const next_node: *Node = &self.nodes[node.neighbor_next.asInt()];

        // Check if the neighbor node can fit the requested size
        if (next_node.data_size < @as(Size, @intCast(size_diff))) {
            return false;
        }
        log.debug("Resizing node {d} from {d} to {d} (resize)", .{
            node_index.asInt(),
            node.data_size,
            (node.data_size + @as(Size, @intCast(size_diff))),
        });
        node.data_size += @as(Size, @intCast(size_diff));

        const remainder_size = (next_node.data_size - @as(Size, @intCast(size_diff)));
        const neighbor_next = next_node.neighbor_next;

        // Remove node from the bin linked list
        self.removeNodeFromBin(node.neighbor_next);

        if (remainder_size > 0) {
            const new_node_index = self.insertNodeIntoBin(remainder_size, (node.data_offset + node.data_size));

            if (neighbor_next != .unused) {
                self.nodes[neighbor_next.asInt()].neighbor_prev = new_node_index;
                self.nodes[new_node_index.asInt()].neighbor_next = neighbor_next;
            }
            self.nodes[new_node_index.asInt()].neighbor_prev = node_index;
            node.neighbor_next = new_node_index;
        }

        assert(next_node.neighbor_prev == Node.Index.from(node_index.asInt()));
        node.neighbor_next = next_node.neighbor_next;
    } else if (size_diff < 0) {
        const remainder_size: Size = @intCast(-size_diff);
        log.debug("Resizing node {d} from {d} to {d} (resize)", .{
            node_index.asInt(),
            node.data_size,
            (node.data_size - @as(Size, @intCast(-size_diff))),
        });
        node.data_size -= remainder_size;

        const new_node_index = self.insertNodeIntoBin(remainder_size, (node.data_offset + node.data_size));

        if (node.neighbor_next != .unused) {
            self.nodes[node.neighbor_next.asInt()].neighbor_prev = new_node_index;
            self.nodes[new_node_index.asInt()].neighbor_next = node.neighbor_next;
        }
        self.nodes[new_node_index.asInt()].neighbor_prev = node_index;
        node.neighbor_next = new_node_index;
    }

    return true;
}

fn innerFree(self: *Self, metadata: InnerAllocation.Metadata) void {
    const node_index = metadata;

    assert(self.is_used.isSet(node_index.asInt()));
    const node: *Node = &self.nodes[node_index.asInt()];

    // Merge with neighbors...
    var offset = node.data_offset;
    var size = node.data_size;

    if (node.neighbor_prev != .unused and !self.is_used.isSet(node.neighbor_prev.asInt())) {
        // Previous (contiguous) free node: Change offset to previous node offset. Sum sizes
        const prev_node: *Node = &self.nodes[node.neighbor_prev.asInt()];
        offset = prev_node.data_offset;
        size += prev_node.data_size;

        // Remove node from the bin linked list and put it in the freelist
        self.removeNodeFromBin(node.neighbor_prev);

        assert(prev_node.neighbor_next == Node.Index.from(node_index.asInt()));
        node.neighbor_prev = prev_node.neighbor_prev;
    }

    if (node.neighbor_next != .unused and self.is_used.isSet(node.neighbor_next.asInt()) == false) {
        // Next (contiguous) free node: Offset remains the same. Sum sizes.
        const next_node: *Node = &self.nodes[node.neighbor_next.asInt()];
        size += next_node.data_size;

        // Remove node from the bin linked list and put it in the freelist
        self.removeNodeFromBin(node.neighbor_next);

        assert(next_node.neighbor_prev == Node.Index.from(node_index.asInt()));
        node.neighbor_next = next_node.neighbor_next;
    }

    const neighbor_next = node.neighbor_next;
    const neighbor_prev = node.neighbor_prev;

    // Insert the removed node to freelist
    self.free_offset.incr();
    log.debug("Putting node {d} into freelist[{d}] (free)", .{ node_index.asInt(), self.free_offset.asInt() });
    self.free_nodes[self.free_offset.asInt()] = node_index;

    // Insert the (combined) free node to bin
    const combined_node_index = self.insertNodeIntoBin(size, offset);

    // Connect neighbors with the new combined node
    if (neighbor_next != .unused) {
        self.nodes[combined_node_index.asInt()].neighbor_next = neighbor_next;
        self.nodes[neighbor_next.asInt()].neighbor_prev = combined_node_index;
    }
    if (neighbor_prev != .unused) {
        self.nodes[combined_node_index.asInt()].neighbor_prev = neighbor_prev;
        self.nodes[neighbor_prev.asInt()].neighbor_next = combined_node_index;
    }
}

/// To conform to Zig's `Allocator` interface this allocator uses
/// embedded metadata and might not be suitable for some use cases.
/// If you need externally stored metadata, use the `...WithMetadata`
/// functions this type provides.
/// Note that the effective size of all allocations with embedded
/// metadata will be at least `@sizeOf(Node.Index) + alloc_size`.
pub fn allocator(self: *Self) Allocator {
    return Allocator{
        .ptr = @ptrCast(self),
        .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = Allocator.noRemap,
            .free = free,
        },
    };
}

fn alloc(context: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    _ = ret_addr;

    const size = math.cast(Size, len) orelse {
        @branchHint(.cold);
        return null;
    };
    const alignment_in_bytes = math.cast(Size, alignment.toByteUnits()) orelse {
        @branchHint(.cold);
        return null;
    };

    const self: *Self = @ptrCast(@alignCast(context));
    const inner_allocation = self.innerAlloc(.embedded, size, alignment_in_bytes);
    if (inner_allocation.isOutOfMemory()) {
        @branchHint(.unlikely);
        return null;
    }

    const ptr = self.ptr[inner_allocation.offset..];
    const metadata_ptr: *InnerAllocation.Metadata =
        @ptrCast(@alignCast(self.ptr[(inner_allocation.offset - @sizeOf(InnerAllocation.Metadata))..][0..@sizeOf(InnerAllocation.Metadata)]));
    metadata_ptr.* = inner_allocation.metadata;

    return ptr;
}

fn resize(context: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
    _ = .{ alignment, ret_addr };

    const old_size: Size = @intCast(memory.len);
    const new_size = math.cast(Size, new_len) orelse {
        @branchHint(.cold);
        return false;
    };

    const self: *Self = @ptrCast(@alignCast(context));

    assert(memory.len >= @sizeOf(InnerAllocation.Metadata));
    assert(self.ownsSlice(memory));

    const metadata_ptr: *InnerAllocation.Metadata = @ptrFromInt(@intFromPtr(memory.ptr) - @sizeOf(InnerAllocation.Metadata));
    assert(self.ownsPtr(@ptrCast(metadata_ptr)));
    const metadata = metadata_ptr.*;
    assert(metadata != .unused);

    return self.innerResize(metadata, old_size, new_size);
}

fn free(context: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
    _ = .{ alignment, ret_addr };

    const self: *Self = @ptrCast(@alignCast(context));

    assert(memory.len >= @sizeOf(InnerAllocation.Metadata));
    assert(self.ownsSlice(memory));

    const metadata_ptr: *InnerAllocation.Metadata = @ptrFromInt(@intFromPtr(memory.ptr) - @sizeOf(InnerAllocation.Metadata));
    assert(self.ownsPtr(@ptrCast(metadata_ptr)));
    const metadata = metadata_ptr.*;
    assert(metadata != .unused);

    self.innerFree(metadata);
}

fn sliceContainsPtr(container: []const u8, ptr: [*]const u8) bool {
    return @intFromPtr(ptr) >= @intFromPtr(container.ptr) and
        @intFromPtr(ptr) < (@intFromPtr(container.ptr) + container.len);
}

fn sliceContainsSlice(container: []const u8, slice: []const u8) bool {
    return @intFromPtr(slice.ptr) >= @intFromPtr(container.ptr) and
        (@intFromPtr(slice.ptr) + slice.len) <= (@intFromPtr(container.ptr) + container.len);
}

const builtin = @import("builtin");
const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const Alignment = mem.Alignment;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const log = std.log.scoped(.suballocator);

// TESTS

const testing_log_level = std.log.Level.debug;

fn testFloatSizeConversion(size: Size, size_floor: Size, size_ceil: Size) !void {
    const floored = floatFromSize(.floor, size);
    const ceiled = floatFromSize(.ceil, size);
    try testing.expect(sizeFromFloat(floored) == size_floor);
    try testing.expect(sizeFromFloat(ceiled) == size_ceil);
}

test "f8<->Size conversion" {
    // From known bins
    try testing.expect(sizeFromFloat(@bitCast(@as(u8, 3))) == 3);
    try testing.expect(sizeFromFloat(@bitCast(@as(u8, 92))) == 12288);
    try testing.expect(sizeFromFloat(@bitCast(@as(u8, 211))) == 369098752);

    // To known bins
    try testing.expect(floatFromSize(.floor, 3).asInt() == 3);
    try testing.expect(floatFromSize(.floor, 11688920).asInt() == 171);
    try testing.expect(floatFromSize(.ceil, 11688920).asInt() == 172);

    // Exact
    for (0..17) |i| {
        const size: Size = @intCast(i);
        try testFloatSizeConversion(size, size, size);
    }
    try testFloatSizeConversion(180224, 180224, 180224);
    try testFloatSizeConversion(2952790016, 2952790016, 2952790016);

    // Between bins
    try testFloatSizeConversion(19, 18, 20);
    try testFloatSizeConversion(21267, 20480, 22528);
    try testFloatSizeConversion(24678495, 23068672, 25165824);
}

test findFirstSetAfter {
    const n: u8 = 0b0001_0010;

    const b1 = findFirstSetAfter(n, 0);
    const b2 = findFirstSetAfter(n, 3);
    const b3 = findFirstSetAfter(n, 4);
    const b4 = findFirstSetAfter(n, 6);

    try testing.expect(b1 != null and b1.? == 1);
    try testing.expect(b2 != null and b2.? == 4);
    try testing.expect(b3 != null and b3.? == 4);
    try testing.expect(b4 == null);
}

var test_memory: [800000 * @sizeOf(u64)]u8 = undefined;
const test_max_allocs = 800000;

test "basic usage with external metadata" {
    testing.log_level = testing_log_level;

    var self = try Self.init(testing.allocator, &test_memory, test_max_allocs);
    defer self.deinit(testing.allocator);
    const ptr = try self.createWithMetadata(u16);
    defer self.destroyWithMetadata(ptr);
    const slice = try self.allocWithMetadata(u16, 3);
    defer self.freeWithMetadata(slice);
    try testing.expect(slice.len == 3);
    const big_slice = try self.allocWithMetadata(u16, 600000);
    defer self.freeWithMetadata(big_slice);
    try testing.expect(big_slice.len == 600000);
}

test "basic usage with embedded metadata" {
    testing.log_level = testing_log_level;

    var self = try Self.init(testing.allocator, &test_memory, test_max_allocs);
    defer self.deinit(testing.allocator);

    const a = self.allocator();

    try std.heap.testAllocator(a);
    try std.heap.testAllocatorAligned(a);
    try std.heap.testAllocatorLargeAlignment(a);
    try std.heap.testAllocatorAlignedShrink(a);
}

test "skewed allocation parameters" {
    testing.log_level = testing_log_level;

    // FIXME use mem.Alignment on zig master
    const buffer = try testing.allocator.alignedAlloc(u8, 16, 1001);
    defer testing.allocator.free(buffer);

    // 1000 bytes capacity
    var self = try Self.init(testing.allocator, buffer[1..buffer.len], 32);
    defer self.deinit(testing.allocator);

    // Since buffer is 15 bytes out of alignment padding is required.
    // This padding should be released back to the allocator as a node.
    const a1 = try self.alignedAllocWithMetadata(u8, .@"16", 500);
    defer self.freeWithMetadata(a1);

    // Should result in two nodes with 15 and 485 bytes respectively.
    // We cannot allocate the full 485 bits here because of fragmentation.
    const a2 = try self.allocWithMetadata(u8, 460);
    defer self.freeWithMetadata(a2);

    const a3 = try self.allocWithMetadata(u8, 15);
    defer self.freeWithMetadata(a3);
}

test "resize" {
    testing.log_level = testing_log_level;

    var self = try Self.init(testing.allocator, &test_memory, test_max_allocs);
    defer self.deinit(testing.allocator);

    const a1 = try self.allocWithMetadata(u8, 12);
    defer self.freeWithMetadata(a1);

    const a2 = try self.allocWithMetadata(u8, 12);

    var a3 = try self.allocWithMetadata(u8, 12);
    defer self.freeWithMetadata(a3);

    self.freeWithMetadata(a2);

    const ok = self.resizeWithMetadata(&a3, 20);
    try testing.expect(ok);
}

test reset {
    testing.log_level = testing_log_level;

    var buffer: [8]u8 = undefined;
    var self = try Self.init(testing.allocator, @ptrCast(&buffer), 32);
    defer self.deinit(testing.allocator);

    var allocation = try self.allocWithMetadata(u8, 8);
    defer self.freeWithMetadata(allocation);

    try testing.expectError(Allocator.Error.OutOfMemory, self.allocWithMetadata(u8, 1));

    self.reset();

    allocation = try self.allocWithMetadata(u8, 8);
}

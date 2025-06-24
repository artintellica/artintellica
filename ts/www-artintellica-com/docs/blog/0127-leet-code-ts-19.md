+++
title = "LeetCode in TypeScript, Part 1.9: Mastering the LRU Cache"
author = "Artintellica"
date = "2025-06-24"
+++

Welcome back to our series on solving LeetCode problems with TypeScript,
tailored for Silicon Valley job interviews! Today, we're diving into a
fascinating and challenging problem: designing and implementing a **Least
Recently Used (LRU) Cache**. This problem is a staple in technical interviews,
especially at top-tier companies like Amazon and Google, as it tests your
ability to combine data structures like hash maps and doubly linked lists into
an efficient design. Whether you're preparing for a coding round or a system
design-adjacent question, mastering the LRU Cache will give you a significant
edge. Let's break it down step by step, explore the problem with an
easy-to-understand analogy, and implement a solution in TypeScript.

## Introduction to the LRU Cache Problem

The LRU Cache problem (LeetCode #146) is classified as Medium to Hard due to its
requirement for a deep understanding of data structure design. The task is to
create a cache with a fixed capacity that stores key-value pairs. When the cache
reaches its capacity, it should evict the least recently used item to make room
for a new one. Additionally, both getting and putting items in the cache should
be performed in $O(1)$ time complexity, which means we need an efficient way to
track usage order and access elements quickly.

This problem is highly relevant for interviews because it mimics real-world
scenarios like caching in web browsers or databases, where quick access and
efficient memory management are critical. It tests your ability to:

- Use a hash map for fast lookups.
- Maintain order with a doubly linked list for quick insertion and deletion.
- Combine multiple data structures into a cohesive design.

In Silicon Valley interviews, especially at companies like Amazon and Google,
you might encounter this as part of a coding round or even as a precursor to a
broader system design discussion. Let's simplify the concept before diving into
the technical details.

## ELI5: Understanding LRU Cache Like a Toy Box

Imagine you have a small toy box that can only hold 3 toys at a time. Every time
you play with a toy, you put it on top of the pile in the box because it's the
"most recently used." If the box is full and you want to add a new toy, you have
to take out the toy at the bottom of the pile—the one you haven't played with in
the longest time (the "least recently used"). To find a toy quickly, you also
have a little notebook where you write down which toy is in which spot in the
box.

In this analogy:

- The toy box is the cache with a fixed capacity.
- The toys are the key-value pairs (data) you're storing.
- The pile order (top to bottom) represents how recently each toy was used.
- The notebook is like a hash map, letting you find a toy instantly without
  digging through the box.
- Adding a new toy or playing with an existing one means updating both the pile
  (order of usage) and the notebook (quick lookup).

The challenge is to make sure that finding a toy, playing with it (moving it to
the top), or replacing the bottom toy with a new one all happen super fast,
without rearranging the whole box every time. That's what we'll solve with our
data structures!

## Designing the LRU Cache: Key Concepts

To achieve $O(1)$ time complexity for both `get` and `put` operations, we need
two main data structures:

1. **Hash Map**: For instant lookups of cache items by key. It will store keys
   mapped to nodes in our linked list.
2. **Doubly Linked List**: To maintain the order of usage. The most recently
   used item will be at the head (front), and the least recently used will be at
   the tail (end). A doubly linked list allows us to move nodes (update order)
   or remove them in $O(1)$ time because each node has pointers to both the
   previous and next nodes.

Here's the strategy:

- When we `get` a key, we find it in the hash map, move its node to the head of
  the linked list (marking it as most recently used), and return its value.
- When we `put` a key-value pair, if the key exists, we update its value and
  move it to the head. If it doesn't exist and the cache is full, we remove the
  tail node (least recently used), update the hash map, and add the new node to
  the head. If there's space, we simply add the new node to the head.
- We use dummy head and tail nodes in the doubly linked list to simplify edge
  cases (like removing the first or last node).

Let's implement this in TypeScript, leveraging type safety to make our code
robust and clear.

## TypeScript Implementation of LRU Cache

We'll start by defining the structure of our doubly linked list node and then
build the LRU Cache class with all required functionality. Below is the complete
solution with detailed comments explaining each part.

```typescript
// Define the structure of a node in the doubly linked list
interface Node {
  key: number;
  value: number;
  prev: Node | null;
  next: Node | null;
}

class LRUCache {
  private capacity: number; // Maximum number of items the cache can hold
  private cache: Map<number, Node>; // Hash map for O(1) lookup of nodes by key
  private dummyHead: Node; // Dummy head node for the doubly linked list
  private dummyTail: Node; // Dummy tail node for the doubly linked list

  constructor(capacity: number) {
    this.capacity = capacity;
    this.cache = new Map<number, Node>();
    // Initialize dummy head and tail to simplify list operations
    this.dummyHead = { key: 0, value: 0, prev: null, next: null };
    this.dummyTail = { key: 0, value: 0, prev: null, next: null };
    this.dummyHead.next = this.dummyTail;
    this.dummyTail.prev = this.dummyHead;
  }

  // Helper method to add a node right after the head
  private addNode(node: Node): void {
    node.prev = this.dummyHead;
    node.next = this.dummyHead.next;
    this.dummyHead.next!.prev = node;
    this.dummyHead.next = node;
  }

  // Helper method to remove an existing node from the linked list
  private removeNode(node: Node): void {
    const prev = node.prev;
    const next = node.next;
    prev!.next = next;
    next!.prev = prev;
  }

  // Helper method to move a node to the head (most recently used)
  private moveToHead(node: Node): void {
    this.removeNode(node);
    this.addNode(node);
  }

  // Get the value associated with a key if it exists in the cache
  get(key: number): number {
    const node = this.cache.get(key);
    if (!node) {
      return -1; // Key not found
    }
    // Move to head (mark as most recently used)
    this.moveToHead(node);
    return node.value;
  }

  // Put a key-value pair into the cache
  put(key: number, value: number): void {
    const node = this.cache.get(key);
    if (node) {
      // If key exists, update its value and move to head
      node.value = value;
      this.moveToHead(node);
    } else {
      // Create a new node
      const newNode: Node = { key, value, prev: null, next: null };
      this.cache.set(key, newNode);
      this.addNode(newNode);

      // If over capacity, remove the least recently used item (tail)
      if (this.cache.size > this.capacity) {
        // Remove from hash map and linked list
        const tail = this.dummyTail.prev!;
        this.cache.delete(tail.key);
        this.removeNode(tail);
      }
    }
  }
}

// Example usage and testing
function testLRUCache() {
  const cache = new LRUCache(2);
  cache.put(1, 1); // Cache is {1=1}
  cache.put(2, 2); // Cache is {1=1, 2=2}
  console.log(cache.get(1)); // Returns 1
  cache.put(3, 3); // Evicts key 2, cache is {1=1, 3=3}
  console.log(cache.get(2)); // Returns -1 (not found)
  cache.put(4, 4); // Evicts key 1, cache is {4=4, 3=3}
  console.log(cache.get(1)); // Returns -1 (not found)
  console.log(cache.get(3)); // Returns 3
  console.log(cache.get(4)); // Returns 4
}

testLRUCache();
```

### Explanation of the Code

- **Node Interface**: Defines the structure of each node in the doubly linked
  list with `key`, `value`, and pointers to `prev` and `next` nodes.
  TypeScript's type safety ensures we don't mix up properties.
- **LRUCache Class**: Contains the main logic with a `Map` for $O(1)$ lookups
  and a doubly linked list for maintaining order.
- **Helper Methods**: `addNode`, `removeNode`, and `moveToHead` manage the
  linked list operations efficiently.
- **get Method**: Retrieves a value by key, updates the usage order, and returns
  -1 if not found.
- **put Method**: Adds or updates a key-value pair, evicts the least recently
  used item if necessary, and maintains the order.
- **Time Complexity**: Both `get` and `put` operations are $O(1)$ because hash
  map lookups and doubly linked list node movements (thanks to direct pointers)
  are constant time.
- **Space Complexity**: $O(capacity)$ to store the key-value pairs in the hash
  map and linked list.

### Why TypeScript Shines Here

TypeScript's `Map` and interface definitions make the code more readable and
less error-prone. For instance, explicitly typing the `cache` as
`Map<number, Node>` ensures that we only store nodes associated with numeric
keys, catching potential bugs during development. This is particularly useful in
a collaborative environment or when maintaining code over time, which aligns
with practices at Silicon Valley companies.

## Conclusion: Key Takeaways from LRU Cache

In this blog post, we've tackled the LRU Cache problem, a medium-to-hard
LeetCode challenge that's a favorite in Silicon Valley interviews at companies
like Amazon and Google. Here's what we learned:

- The LRU Cache requires $O(1)$ time complexity for operations, achieved by
  combining a hash map (for lookups) and a doubly linked list (for order
  maintenance).
- We broke down the concept using a toy box analogy, making it clear how "most
  recently used" and "least recently used" items are managed.
- Our TypeScript implementation leverages type safety with interfaces and
  explicit typing, ensuring robust and maintainable code.
- Key operations like `get` and `put` were implemented with helper methods to
  manage the doubly linked list efficiently.

Mastering the LRU Cache not only prepares you for coding interviews but also
deepens your understanding of data structure design—a critical skill for
real-world applications. In the next post, we'll explore another challenging
problem to further build your algorithmic toolkit. Until then, keep practicing,
and happy coding!

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

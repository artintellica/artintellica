class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}

function reverseListIterative(head: ListNode | null): ListNode | null {
  let prev: ListNode | null = null; // Start with no previous node
  let curr: ListNode | null = head; // Start at the head of the list

  while (curr !== null) {
    // Store the next node temporarily so we don't lose it
    const nextTemp: ListNode | null = curr.next;
    // Reverse the link: point current node's next to the previous node
    curr.next = prev;
    // Move prev to current node for the next iteration
    prev = curr;
    // Move current to the next node (stored in nextTemp)
    curr = nextTemp;
  }

  // At the end, prev is the new head of the reversed list
  return prev;
}

function reverseListRecursive(head: ListNode | null): ListNode | null {
  // Base case: if head is null or it's the last node, return head
  if (head === null || head.next === null) {
    return head;
  }

  // Recursively traverse to the end of the list
  const newHead: ListNode | null = reverseListRecursive(head.next);
  // Reverse the link: make the next node point back to current node
  head.next.next = head;
  // Break the forward link: set current node's next to null
  head.next = null;

  // Return the new head (last node becomes first)
  return newHead;
}

// Helper function to create a linked list from an array
function createLinkedList(arr: number[]): ListNode | null {
  if (arr.length === 0) {
    return null;
  }
  const head = new ListNode(arr[0]);
  let current = head;
  for (let i = 1; i < arr.length; i++) {
    current.next = new ListNode(arr[i]);
    current = current.next;
  }
  return head;
}

// Helper function to print a linked list
function printLinkedList(head: ListNode | null): void {
  const result: number[] = [];
  let current = head;
  while (current !== null) {
    result.push(current.val);
    current = current.next;
  }
  console.log(result.join(" -> "));
}

// Test the solutions
let list = createLinkedList([1, 2, 3, 4, 5]);
console.log("Original List:");
printLinkedList(list);

// Test Iterative Solution
const reversedIterative = reverseListIterative(list);
console.log("Reversed List (Iterative):");
printLinkedList(reversedIterative);

// Recreate list for recursive test
list = createLinkedList([1, 2, 3, 4, 5]);
const reversedRecursive = reverseListRecursive(list);
console.log("Reversed List (Recursive):");
printLinkedList(reversedRecursive);

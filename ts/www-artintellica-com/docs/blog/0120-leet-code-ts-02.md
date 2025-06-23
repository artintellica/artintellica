+++
title = "LeetCode in TypeScript, Part 1.2: Reversing a Singly Linked List"
author = "Artintellica"
date = "2025-06-23"
+++

Welcome back to our LeetCode series where we dive into common interview problems
using TypeScript! Today, we’re tackling a classic and fundamental problem:
**Reverse Linked List**. This problem is labeled as "Easy" on LeetCode, but
don’t let that fool you—it tests your understanding of pointer manipulation and
data structure traversal, skills that are critical in technical interviews,
especially at companies like Meta and Microsoft. Whether you're a beginner or
brushing up for a Silicon Valley interview, mastering linked lists is a must. In
this post, we’ll break down the problem, explain it in simple terms, provide a
TypeScript solution, and explore both iterative and recursive approaches.

## Introduction to Reverse Linked List

The problem is straightforward: given a singly linked list, reverse the order of
its nodes. A singly linked list is a linear data structure where each node
points to the next node in the sequence, and the last node points to null. For
example, if the input list is `1 -> 2 -> 3 -> 4 -> 5`, the output should be
`5 -> 4 -> 3 -> 2 -> 1`. This task requires us to rewire the pointers of each
node so that they point backward instead of forward.

Why is this relevant for interviews? Linked lists are a foundational data
structure, and reversing one demonstrates your ability to manipulate pointers,
handle edge cases (like an empty list or a single node), and think through
iterative or recursive logic. It’s a common question in early interview rounds
to assess basic problem-solving skills.

The key concepts here are:

- **Linked Lists**: Understanding how nodes are connected via pointers.
- **Iteration vs. Recursion**: Two distinct ways to solve the problem, each with
  its own trade-offs in terms of readability and space complexity.

Let’s simplify this further with an ELI5 explanation before diving into the
code.

## ELI5: Reversing a Linked List

Imagine you have a line of toy cars, each connected to the next with a string,
so they can only move forward. Car 1 is tied to Car 2, Car 2 to Car 3, and so
on, up to Car 5. Now, your job is to make them line up in reverse order—Car 5
first, then Car 4, down to Car 1—without breaking the connections.

Here’s how you might do it: start at Car 1, untie the string to Car 2, and tie a
new string from Car 2 to Car 1. Then move to Car 2, untie the string to Car 3,
and tie Car 3 to Car 2. Keep doing this until you’ve flipped all the
connections. In the end, Car 5 will be at the front, and each car points to the
one that was originally in front of it.

That’s what reversing a linked list is like! Each “car” is a node, and the
“string” is a pointer. We’re just changing who points to whom, step by step,
until the whole line is flipped.

## TypeScript Solution: Problem Setup

Before we write the solution, let’s define the structure of a linked list node
in TypeScript. We’ll use a class to represent a node, which has a value and a
pointer to the next node.

```typescript
class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}
```

The problem on LeetCode is defined as:

- **Input**: Head of a singly linked list (type `ListNode | null`).
- **Output**: Head of the reversed linked list (type `ListNode | null`).

We’ll solve this in two ways: iteratively and recursively. Both approaches
achieve the same result, but they differ in implementation and space complexity.
Let’s start with the iterative solution, which is often preferred in interviews
for its efficiency.

## Solution 1: Iterative Approach

The iterative approach uses a loop to traverse the list and reverse the pointers
one by one. We maintain three pointers: `prev` (previous node), `curr` (current
node), and `nextTemp` (to temporarily store the next node before we break the
link).

Here’s the full solution with detailed comments:

```typescript
function reverseListIterative(head: ListNode | null): ListNode | null {
  let prev: ListNode | null = null; // Start with no previous node
  let curr: ListNode | null = head; // Start at the head of the list

  while (curr !== null) {
    // Store the next node temporarily so we don't lose it
    let nextTemp: ListNode | null = curr.next;
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
```

### How It Works

- Initially, `prev` is `null`, and `curr` points to the head.
- For each node, we save its `next` in `nextTemp`, then set its `next` to `prev`
  (reversing the link).
- We update `prev` to the current node and move `curr` to `nextTemp`.
- When `curr` becomes `null`, `prev` is the new head of the reversed list.

### Complexity

- **Time Complexity**: $O(n)$, where $n$ is the number of nodes in the list. We
  traverse the list once.
- **Space Complexity**: $O(1)$, as we only use a constant amount of extra space
  regardless of list size.

## Solution 2: Recursive Approach

The recursive approach breaks the problem into smaller subproblems. We
recursively traverse to the end of the list, then reverse the pointers as we
unwind the call stack.

Here’s the full solution with comments:

```typescript
function reverseListRecursive(head: ListNode | null): ListNode | null {
  // Base case: if head is null or it's the last node, return head
  if (head === null || head.next === null) {
    return head;
  }

  // Recursively traverse to the end of the list
  let newHead: ListNode | null = reverseListRecursive(head.next);
  // Reverse the link: make the next node point back to current node
  head.next.next = head;
  // Break the forward link: set current node's next to null
  head.next = null;

  // Return the new head (last node becomes first)
  return newHead;
}
```

### How It Works

- The recursion goes to the last node (base case), which becomes the new head.
- For each recursive call unwinding, we reverse the link by setting
  `head.next.next = head` (e.g., if `head` is 4 and `head.next` is 5, set 5’s
  next to 4).
- We set `head.next = null` to break the original forward link.
- The new head propagates back up the call stack.

### Complexity

- **Time Complexity**: $O(n)$, as we traverse the list once.
- **Space Complexity**: $O(n)$, due to the recursive call stack, which can be a
  drawback in interviews if space is a concern.

## Testing the Solutions

Let’s write a quick test to verify both solutions work. We’ll create a linked
list `1 -> 2 -> 3 -> 4 -> 5`, reverse it, and print the result.

```typescript
// Helper function to create a linked list from an array
function createLinkedList(arr: number[]): ListNode | null {
  if (arr.length === 0) return null;
  let head = new ListNode(arr[0]);
  let current = head;
  for (let i = 1; i < arr.length; i++) {
    current.next = new ListNode(arr[i]);
    current = current.next;
  }
  return head;
}

// Helper function to print a linked list
function printLinkedList(head: ListNode | null): void {
  let result: number[] = [];
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
let reversedIterative = reverseListIterative(list);
console.log("Reversed List (Iterative):");
printLinkedList(reversedIterative);

// Recreate list for recursive test
list = createLinkedList([1, 2, 3, 4, 5]);
let reversedRecursive = reverseListRecursive(list);
console.log("Reversed List (Recursive):");
printLinkedList(reversedRecursive);
```

### Output

```
Original List:
1 -> 2 -> 3 -> 4 -> 5
Reversed List (Iterative):
5 -> 4 -> 3 -> 2 -> 1
Reversed List (Recursive):
5 -> 4 -> 3 -> 2 -> 1
```

## Interview Tips and Edge Cases

When solving this problem in an interview:

- **Clarify Input**: Confirm if the list could be empty (`null`) or have a
  single node. Both are handled by our solutions.
- **Discuss Trade-offs**: Mention that the iterative solution is more
  space-efficient ($O(1)$ space) compared to recursive ($O(n)$ space).
- **Draw Diagrams**: Visualize the pointer changes on a whiteboard—it helps
  avoid mistakes and shows your thought process.
- **Edge Cases**: Test with an empty list (`null`), a single node (`1 -> null`),
  and a longer list. Our code handles all these cases.

## Conclusion

In this blog post, we explored the **Reverse Linked List** problem, a staple in
technical interviews at companies like Meta and Microsoft. We broke down the
concept using an ELI5 analogy of flipping toy cars in a line, making the pointer
manipulation intuitive. We provided two complete TypeScript solutions—iterative
and recursive—each with detailed explanations, complexity analysis, and test
code to verify the results. Key takeaways include:

- Linked lists require careful pointer management to avoid losing nodes.
- The iterative approach is space-efficient ($O(1)$ space), while the recursive
  approach is more concise but uses $O(n)$ space.
- Understanding both methods equips you to discuss trade-offs in an interview
  setting.

Stay tuned for the next post in our LeetCode series, where we’ll tackle more
linked list problems or move on to arrays and strings. Practice this problem
with different inputs, and don’t hesitate to draw out the steps—it’s a great way
to solidify your understanding! If you have questions or alternative solutions,
drop them in the comments. Happy coding!

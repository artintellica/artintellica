// Definition for singly-linked list node
class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}

function mergeTwoLists(
  list1: ListNode | null,
  list2: ListNode | null,
): ListNode | null {
  // Create a dummy node to serve as the head of the merged list
  const dummy = new ListNode(0);
  // Use a current pointer to build the merged list
  let current = dummy;

  // Traverse both lists while neither is empty
  while (list1 !== null && list2 !== null) {
    // Compare values of the current nodes from list1 and list2
    if (list1.val <= list2.val) {
      // If list1's value is smaller or equal, add it to the merged list
      current.next = list1;
      // Move to the next node in list1
      list1 = list1.next;
    } else {
      // If list2's value is smaller, add it to the merged list
      current.next = list2;
      // Move to the next node in list2
      list2 = list2.next;
    }
    // Move the current pointer forward
    current = current.next;
  }

  // If list1 still has nodes, append them to the merged list
  if (list1 !== null) {
    current.next = list1;
  }
  // If list2 still has nodes, append them to the merged list
  if (list2 !== null) {
    current.next = list2;
  }

  // Return the head of the merged list (skip the dummy node)
  return dummy.next;
}

// Helper function to create a linked list from an array (for testing)
function arrayToList(arr: number[]): ListNode | null {
  if (arr.length === 0) {
    return null;
  }
  const dummy = new ListNode(0);
  let current = dummy;
  for (const val of arr) {
    current.next = new ListNode(val);
    current = current.next;
  }
  return dummy.next;
}

// Helper function to convert linked list back to array (for output)
function listToArray(list: ListNode | null): number[] {
  const result: number[] = [];
  let current = list;
  while (current !== null) {
    result.push(current.val);
    current = current.next;
  }
  return result;
}

// Test the merge function
const list1 = arrayToList([1, 2, 4]);
const list2 = arrayToList([1, 3, 4]);
const merged = mergeTwoLists(list1, list2);
console.log(listToArray(merged)); // Output: [1, 1, 2, 3, 4, 4]

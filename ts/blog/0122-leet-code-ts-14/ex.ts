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
  if (list1 == null && list2 == null) {
    return null;
  }
  if (list1 == null) {
    return list2;
  }
  if (list2 == null) {
    return list1;
  }

  const dummy: ListNode = new ListNode(0);
  let current = dummy;

  while (list1 !== null && list2 !== null) {
    if (list1.val <= list2.val) {
      current.next = list1;
      list1 = list1.next;
    } else {
      current.next = list2;
      list2 = list2.next;
    }
    current = current.next;
  }

  if (list1 !== null) {
    current.next = list1;
  }
  if (list2 !== null) {
    current.next = list2;
  }

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

class ListNode {
  val: number;
  next?: ListNode;

  constructor(val: number, next?: ListNode) {
    this.val = val;
    this.next = next;
  }
}

function createList(n: number, max: number): ListNode {
  if (n < max) {
    const node = new ListNode(n);
    node.next = createList(n + 1, max);
    return node;
  }
  const node = new ListNode(n);
  return node;
}

function printList(node: ListNode | undefined) {
  if (!node) {
    return "";
  }
  if (node.next) {
    return `${node.val} -> ${printList(node.next)}`;
  }
  return `${node.val}`;
}

const list = createList(0, 5);
console.log(printList(list));

function reverseListLoop(node: ListNode) {
  let prev: ListNode | undefined = undefined;
  let curr: ListNode | undefined = node;
  while (curr) {
    const next: ListNode | undefined = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }
  return prev;
}

console.log(printList(reverseListLoop(createList(0, 5))));


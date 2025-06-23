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

function printList(node: ListNode) {
  if (node.next) {
    return `${node.val} -> ${printList(node.next)}`;
  }
  return `${node.val}`;
}

const list = createList(0, 5);
console.log(printList(list));

function reverseListLoop(node: ListNode) {
  let curNode: ListNode | undefined = node;
  let nextNode = curNode.next
  curNode.next = undefined
  let prevNode: ListNode | undefined = undefined
  while (nextNode) {
    curNode.next = prevNode

    prevNode = curNode
    curNode = nextNode
    nextNode = nextNode.next
  }
  return curNode
}

console.log(printList(reverseListLoop(list)))

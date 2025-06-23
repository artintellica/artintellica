class ListNode {
  val: number;
  next?: ListNode;

  constructor(val: number, next?: ListNode) {
    this.val = val;
    this.next = next;
  }
}

function createList(n: number): ListNode {
  if (n > 1) {
    const node = new ListNode(n);
    node.next = createList(n - 1);
    return node;
  }
  const node = new ListNode(n);
  return node;
}

// function printList(node: ListNode) {
//   let str = "";
//   let curNode = node;
// }

const list = createList(5);
console.log(list)

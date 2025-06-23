+++
title = "LeetCode in TypeScript, Part 1.3: Solving Valid Parentheses"
author = "Artintellica"
date = "2025-06-23"
+++

Welcome back to our series on tackling LeetCode problems with TypeScript! In
this post, we're diving into a classic easy-level problem called **Valid
Parentheses**. This problem is a fantastic way to explore fundamental concepts
like stacks and string manipulation, which are often tested in technical
interviews, especially for entry-level roles at companies like Apple. Whether
you're new to coding or brushing up for a Silicon Valley interview, this problem
offers a great opportunity to build your skills. Let's break it down step by
step with TypeScript, including type safety to make our solution robust and
maintainable.

## Introduction to Valid Parentheses

The Valid Parentheses problem is straightforward but tests your ability to
handle structured data and logic. Here's the problem statement from LeetCode:

> Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
> determine if the input string is valid. An input string is valid if:
>
> 1. Open brackets must be closed by the same type of brackets.
> 2. Open brackets must be closed in the correct order.
> 3. Every close bracket has a corresponding open bracket of the same type.

For example:

- `"()"` is valid.
- `"()[]{}"` is valid.
- `"(]"` is not valid.
- `"([)]"` is not valid.
- `"{[]}"` is valid.

This problem is relevant because it tests your understanding of stacks—a key
data structure for tracking order and pairing elements. In interviews, it's
often used to assess how you approach problems that require matching or
balancing, which are common in real-world applications like parsing code or
validating expressions.

## ELI5: Understanding Valid Parentheses

Imagine you're organizing a set of toy boxes that come in pairs: round boxes
(like `()`), square boxes (`[]`), and curly boxes (`{}`). Each box has a lid
that must match it perfectly, and you can only close a box if it's the last one
you opened. Your job is to check if a sequence of opening and closing boxes
makes sense.

- If you open a round box `(` , you must close it with a `)` before closing any
  other box that was opened later.
- If you try to close a box with the wrong lid (like closing a round box with a
  square lid `]`), that's wrong.
- If you have lids left over without boxes to close, or boxes without lids,
  that's also wrong.

To solve this, you can keep a "stack" of opened boxes. Every time you open a
box, you put it on top of the stack. When you close a box, you check if the top
of the stack matches the lid. If it does, you remove the box from the stack. If
it doesn't match, or if there's no box to close, it's invalid. At the end, if
the stack is empty, everything matched perfectly!

## Solving Valid Parentheses with TypeScript

Let's implement a solution in TypeScript. We'll use a stack to keep track of
opening brackets and ensure they match with closing brackets in the correct
order. TypeScript's type system will help us define clear structures and avoid
errors.

### Approach

1. Create a mapping of closing brackets to their corresponding opening brackets.
2. Use a stack (implemented as an array) to store opening brackets as we
   encounter them.
3. For each character in the string:
   - If it's an opening bracket, push it onto the stack.
   - If it's a closing bracket, check if the stack's top element matches it. If
     it does, pop the stack; if not, return false.
4. At the end, check if the stack is empty (all brackets matched).

### TypeScript Code Solution

Here's the complete solution with detailed comments:

```typescript
function isValid(s: string): boolean {
  // Define a mapping of closing brackets to opening brackets using a Map
  const bracketsMap = new Map<string, string>([
    [")", "("],
    ["}", "{"],
    ["]", "["],
  ]);

  // Stack to store opening brackets
  const stack: string[] = [];

  // Iterate through each character in the input string
  for (const char of s) {
    // If the character is a closing bracket (exists as a key in the map)
    if (bracketsMap.has(char)) {
      // Check if stack is empty (no opening bracket to match) or
      // if the top of the stack doesn't match the expected opening bracket
      if (
        stack.length === 0 ||
        stack[stack.length - 1] !== bracketsMap.get(char)
      ) {
        return false;
      }
      // If it matches, pop the top opening bracket from the stack
      stack.pop();
    } else {
      // If it's an opening bracket, push it onto the stack
      stack.push(char);
    }
  }

  // Return true only if all brackets are matched (stack is empty)
  return stack.length === 0;
}

// Test cases to demonstrate the solution
console.log(isValid("()")); // true
console.log(isValid("()[]{}")); // true
console.log(isValid("(]")); // false
console.log(isValid("([)]")); // false
console.log(isValid("{[]}")); // true
```

### Explanation of the Code

- **Type Safety**: We use TypeScript's type annotations (`string` for input,
  `boolean` for output, and `string[]` for the stack) to ensure clarity and
  catch potential type errors early.
- **Map for Matching**: A `Map` stores the closing-to-opening bracket pairs,
  making lookups efficient and readable.
- **Stack Logic**: We use an array as a stack. For each closing bracket, we
  check the top of the stack. If it doesn't match or the stack is empty, the
  string is invalid.
- **Time and Space Complexity**:
  - Time Complexity: $O(n)$, where $n$ is the length of the input string, as we
    traverse the string once.
  - Space Complexity: $O(n)$ in the worst case, where the stack stores up to
    half the string (all opening brackets).

### Why Use TypeScript?

While this problem can be solved in any language, TypeScript adds value by:

- Ensuring the input is typed as `string`, preventing accidental non-string
  inputs.
- Making the code self-documenting with explicit types for the stack and return
  value.
- Allowing IDEs to provide better autocompletion and error checking, which is
  helpful during interviews if you're coding in a TypeScript-friendly
  environment.

## Additional Exercise: Extended Validation

Let's consider a slight variation of the problem as an additional exercise.
Suppose we want to validate not just brackets but also other paired characters,
like angle brackets `<` and `>`. Can we modify our solution to handle custom
pairs?

### Solution to Extended Validation

Here's the modified TypeScript code to handle custom pairs:

```typescript
function isValidExtended(s: string, pairs: [string, string][]): boolean {
  // Create a map from closing to opening characters based on provided pairs
  const charMap = new Map<string, string>();
  for (const [open, close] of pairs) {
    charMap.set(close, open);
  }

  const stack: string[] = [];
  const openingChars = new Set(pairs.map((pair) => pair[0]));

  for (const char of s) {
    if (charMap.has(char)) {
      if (stack.length === 0 || stack[stack.length - 1] !== charMap.get(char)) {
        return false;
      }
      stack.pop();
    } else if (openingChars.has(char)) {
      stack.push(char);
    } else {
      // Ignore characters not in the pairs
      continue;
    }
  }

  return stack.length === 0;
}

// Test cases for extended validation
const bracketPairs: [string, string][] = [
  ["(", ")"],
  ["{", "}"],
  ["[", "]"],
  ["<", ">"],
];
console.log(isValidExtended("()<>", bracketPairs)); // true
console.log(isValidExtended("(<>)", bracketPairs)); // true
console.log(isValidExtended("(<)", bracketPairs)); // false
console.log(isValidExtended("abc()def", bracketPairs)); // true (ignores non-paired chars)
```

### Explanation of Extended Solution

- **Custom Pairs**: We accept an array of tuples defining opening and closing
  pairs, making the function reusable for different sets of characters.
- **Set for Opening Chars**: A `Set` quickly checks if a character is an opening
  character from the provided pairs.
- **Flexibility**: Non-paired characters are ignored, simulating real-world
  parsing where irrelevant data might be present.

This variation shows how the core stack-based approach can be generalized, a
useful skill in interviews where follow-up questions often test adaptability.

## Conclusion

In this blog post, we've tackled the **Valid Parentheses** problem from LeetCode
using TypeScript. We explored how to use a stack to validate bracket sequences,
ensuring that brackets are properly matched and closed in the correct order. Key
takeaways include:

- The importance of stacks for tracking order in problems involving pairing or
  nesting.
- How TypeScript's type system adds clarity and safety to our code, even for
  simple problems.
- A time complexity of $O(n)$ and space complexity of $O(n)$, which are critical
  to discuss in interviews.
- An extended exercise showing how to generalize the solution for custom
  character pairs, demonstrating adaptability.

This problem is a staple in technical interviews, especially at companies like
Apple, because it tests fundamental concepts like stack usage and string parsing
in a concise way. By solving it in TypeScript, we've also highlighted how modern
JavaScript practices can enhance code quality. In the next post, we'll move on
to another easy problem to build on these skills. Stay tuned, and happy coding!

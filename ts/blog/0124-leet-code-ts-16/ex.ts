function lengthOfLongestSubstring(str: string): number {
  let leftIdx = 0;
  let rightIdx = 0;
  let maxLen = 0;
  const charInWindow = new Map<string, number>(); // char -> index

  for (let idx = 0; idx < str.length; idx++) {
    rightIdx = idx;
    const char = str[idx] as string;
    if (charInWindow.has(char)) {
      const newIdx = charInWindow.get(char) as number;
      leftIdx = newIdx;
      // charInWindow.delete(char);
    }
    charInWindow.set(char, idx);
    const newLen = rightIdx - leftIdx;
    if (newLen > maxLen) {
      maxLen = newLen;
    }
  }

  return maxLen;
}

function runTests(): void {
  const testCases: { input: string; expected: number }[] = [
    { input: "abcabcbb", expected: 3 }, // "abc"
    { input: "bbbbb", expected: 1 }, // "b"
    { input: "pwwkew", expected: 3 }, // "wke"
    { input: "", expected: 0 }, // empty string
    { input: "dvdf", expected: 3 }, // "vdf"
  ];

  testCases.forEach((test, index) => {
    const result: number = lengthOfLongestSubstring(test.input);
    console.log(
      `Test ${index + 1}: Input="${test.input}", Expected=${test.expected}, Got=${result}, ${
        result === test.expected ? "PASS" : "FAIL"
      }`,
    );
  });
}

// Run the tests
runTests();

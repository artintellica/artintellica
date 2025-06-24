function wordBreakWithList(s: string, wordDict: string[]): string[] | null {
  // Convert wordDict to a Set for O(1) lookup time
  const wordSet = new Set<string>(wordDict);

  // Create a DP array where dp[i] stores the list of words for s[0...i-1]
  // If no valid segmentation, store null
  const dp: (string[] | null)[] = new Array(s.length + 1).fill(null);

  // Empty string has an empty list of words
  dp[0] = [];

  // Iterate through each position i in the string
  for (let i = 1; i <= s.length; i++) {
    // Check every possible substring ending at i by looking at previous positions j
    for (let j = 0; j < i; j++) {
      if (dp[j] !== null) {
        const word = s.substring(j, i);
        if (wordSet.has(word)) {
          // If this is the first valid segmentation found, or we want to store it
          dp[i] = [...(dp[j] as string[]), word];
          break; // Take the first valid segmentation for simplicity
        }
      }
    }
  }

  // Return the list of words if the entire string can be segmented, else null
  return dp[s.length] as string[] | null;
}

// Test cases to demonstrate the solution with word list output
function runTestsWithList(): void {
  console.log("Test 1:", wordBreakWithList("leetcode", ["leet", "code"])); // Expected: ["leet", "code"]
  console.log("Test 2:", wordBreakWithList("applepenapple", ["apple", "pen"])); // Expected: ["apple", "pen", "apple"]
  console.log(
    "Test 3:",
    wordBreakWithList("catsandog", ["cats", "dog", "sand", "and", "cat"]),
  ); // Expected: null
}

// Run the tests
runTestsWithList();

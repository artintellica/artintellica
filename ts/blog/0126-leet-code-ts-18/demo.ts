function wordBreak(s: string, wordDict: string[]): boolean {
  // Convert wordDict to a Set for O(1) lookup time
  const wordSet = new Set<string>(wordDict);

  // Create a DP array where dp[i] means s[0...i-1] can be segmented
  const dp: boolean[] = new Array(s.length + 1).fill(false);

  // Empty string is always valid
  dp[0] = true;

  // Iterate through each position i in the string
  for (let i = 1; i <= s.length; i++) {
    // Check every possible substring ending at i by looking at previous positions j
    for (let j = 0; j < i; j++) {
      // If s[0...j-1] is valid and s[j...i-1] is in the dictionary
      if (dp[j] && wordSet.has(s.substring(j, i))) {
        dp[i] = true;
        break; // No need to check further once we find a valid split
      }
    }
  }

  // Return whether the entire string can be segmented
  return dp[s.length] as boolean;
}

// Test cases to demonstrate the solution
function runTests(): void {
  console.log("Test 1:", wordBreak("leetcode", ["leet", "code"])); // Expected: true
  console.log("Test 2:", wordBreak("applepenapple", ["apple", "pen"])); // Expected: true
  console.log(
    "Test 3:",
    wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]),
  ); // Expected: false
}

// Run the tests
runTests();

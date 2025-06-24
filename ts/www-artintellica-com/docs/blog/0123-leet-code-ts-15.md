+++
title = "LeetCode in TypeScript, Part 5: Best Time to Buy and Sell Stock"
author = "Artintellica"
date = "2025-06-23"
code = "https://github.com/artintellica/artintellica/tree/main/ts/blog/0123-leet-code-ts-15"
+++

Welcome back to our LeetCode journey with TypeScript! In this post, we're diving
into a classic problem that frequently appears in technical interviews,
especially at companies like Google and in hedge fund tech roles: **Best Time to
Buy and Sell Stock**. This problem tests your ability to work with arrays and
apply a greedy algorithm to maximize profit. Whether you're preparing for a
Silicon Valley interview or just sharpening your algorithmic skills, this
problem offers valuable insights into efficient array traversal. Let's break it
down step by step, explore the concept with an ELI5 analogy, and implement a
solution in TypeScript.

## Introduction to Best Time to Buy and Sell Stock

The problem is straightforward: you're given an array of stock prices where each
element represents the price of a stock on a particular day. Your goal is to
maximize your profit by buying the stock on one day and selling it on a later
day. If no profit is possible, you should return 0.

- **Input**: An array of numbers representing daily stock prices, e.g.,
  `[7,1,5,3,6,4]`.
- **Output**: The maximum profit you can make, e.g., for the input above, the
  maximum profit is 5 (buy at 1, sell at 6).
- **Constraints**: You must buy before selling, and you can only buy and sell
  once. If no profit is possible, return 0.

This problem falls into the "Easy to Medium" difficulty range on LeetCode
because while the concept is simple, identifying the optimal strategy requires
understanding how to minimize the buying price while maximizing the selling
price. It introduces the greedy algorithm approach, where at each step, we make
a locally optimal choice hoping it leads to a globally optimal solution.

## ELI5: Understanding the Problem Like You're Five

Imagine you're at a candy store every day for a week, and each day the price of
your favorite candy changes. You have a little bit of money, and you want to
make the most extra money by buying the candy on a cheap day and selling it to
your friend on a day when it's more expensive. But you can only buy once and
sell once, and you have to sell after you buy.

- On Monday, candy costs 7 coins.
- On Tuesday, it’s only 1 coin (super cheap!).
- On Wednesday, it’s 5 coins.
- On Thursday, it’s 3 coins.
- On Friday, it’s 6 coins.
- On Saturday, it’s 4 coins.

If you buy on Tuesday for 1 coin and sell on Friday for 6 coins, you make 5
coins of profit. That’s the best you can do! If the prices only went down every
day, you’d make 0 profit because you wouldn’t sell at a loss. So, your job is to
look at all the days, find the cheapest day to buy, and the most expensive day
after that to sell, to make the most money.

## Key Concepts: Arrays and Greedy Algorithms

Before we jump into the code, let's clarify the key concepts this problem tests:

- **Arrays**: The input is a simple list of numbers, and we need to traverse it
  efficiently to compare prices.
- **Greedy Algorithm**: Instead of checking every possible buy-sell pair (which
  would be slow with a time complexity of $O(n^2)$), we use a greedy approach.
  We keep track of the minimum price seen so far (the best time to buy) and
  update the maximum profit if selling at the current price yields a better
  result. This reduces the time complexity to $O(n)$, where $n$ is the length of
  the array.

The idea is to iterate through the array once, maintaining two variables:

- The lowest price seen so far (to buy at the cheapest point).
- The maximum profit possible (updated by checking if selling at the current
  price beats the previous best profit).

## TypeScript Solution: Best Time to Buy and Sell Stock

Let's implement the solution in TypeScript. We'll define the function with
proper type annotations to ensure clarity and type safety, which is one of the
benefits of using TypeScript over plain JavaScript.

```typescript
function maxProfit(prices: number[]): number {
  // If the array is empty or has only one element, no profit is possible
  if (prices.length < 2) {
    return 0;
  }

  // Initialize variables:
  // minPrice is the lowest price seen so far (best time to buy)
  // maxProfit is the maximum profit we can make
  let minPrice: number = prices[0];
  let maxProfit: number = 0;

  // Iterate through the array starting from the second element
  for (let i = 1; i < prices.length; i++) {
    const currentPrice = prices[i];

    // Update minPrice if the current price is lower
    if (currentPrice < minPrice) {
      minPrice = currentPrice;
    }
    // Update maxProfit if selling at current price gives a better profit
    else {
      const potentialProfit = currentPrice - minPrice;
      maxProfit = Math.max(maxProfit, potentialProfit);
    }
  }

  return maxProfit;
}
```

### How It Works

- We first check if the array has fewer than 2 elements. If so, no transaction
  is possible, so we return 0.
- We initialize `minPrice` as the first price in the array (the first possible
  buying point) and `maxProfit` as 0 (no profit yet).
- We loop through the array starting from the second element:
  - If the current price is lower than `minPrice`, we update `minPrice` because
    buying cheaper is always better.
  - If the current price is higher, we calculate the potential profit by
    subtracting `minPrice` from the current price and update `maxProfit` if this
    profit is larger than the previous maximum.
- Finally, we return `maxProfit`.

### Example Walkthrough

Let's test it with the example `[7,1,5,3,6,4]`:

- Day 0: Price = 7, minPrice = 7, maxProfit = 0
- Day 1: Price = 1, minPrice = 1 (updated), maxProfit = 0
- Day 2: Price = 5, minPrice = 1, maxProfit = 4 (5-1=4)
- Day 3: Price = 3, minPrice = 1, maxProfit = 4 (3-1=2, no update)
- Day 4: Price = 6, minPrice = 1, maxProfit = 5 (6-1=5, updated)
- Day 5: Price = 4, minPrice = 1, maxProfit = 5 (4-1=3, no update)
- Result: maxProfit = 5

### Time and Space Complexity

- **Time Complexity**: $O(n)$, where $n$ is the length of the prices array. We
  only traverse the array once.
- **Space Complexity**: $O(1)$, as we only use two variables regardless of input
  size.

### Test Cases

Let's run a few test cases to verify our solution:

```typescript
console.log(maxProfit([7, 1, 5, 3, 6, 4])); // Output: 5 (buy at 1, sell at 6)
console.log(maxProfit([7, 6, 4, 3, 1])); // Output: 0 (no profit possible)
console.log(maxProfit([1, 2])); // Output: 1 (buy at 1, sell at 2)
console.log(maxProfit([2])); // Output: 0 (can't sell)
```

## Bonus Exercise: Best Time to Buy and Sell Stock II

As a bonus, let's tackle a related problem often asked in interviews: **Best
Time to Buy and Sell Stock II**. Here, you can buy and sell multiple times, as
long as you sell before buying again. The goal is still to maximize profit.

- **Input**: An array of stock prices, e.g., `[7,1,5,3,6,4]`.
- **Output**: The maximum profit, e.g., for the input above, the maximum profit
  is 7 (buy at 1, sell at 5 for 4 profit; buy at 3, sell at 6 for 3 profit;
  total = 7).

### TypeScript Solution for Best Time to Buy and Sell Stock II

```typescript
function maxProfitII(prices: number[]): number {
  // If the array is empty or has only one element, no profit is possible
  if (prices.length < 2) {
    return 0;
  }

  // Initialize total profit
  let totalProfit: number = 0;

  // Iterate through the array
  for (let i = 1; i < prices.length; i++) {
    // If current price is higher than previous day's price,
    // we can make a profit by buying yesterday and selling today
    if (prices[i] > prices[i - 1]) {
      totalProfit += prices[i] - prices[i - 1];
    }
  }

  return totalProfit;
}
```

### How It Works

- Since we can buy and sell multiple times, we take advantage of every upward
  price movement.
- We iterate through the array, and whenever the current price is higher than
  the previous day's price, we add the difference to our profit (effectively
  buying at the lower price and selling at the higher price).
- This greedy approach captures all possible profits because every increase in
  price can be a separate transaction.

### Example Walkthrough for `[7,1,5,3,6,4]`

- Day 1: Price 1 vs 7, no profit (price decreased)
- Day 2: Price 5 vs 1, profit += 4 (5-1)
- Day 3: Price 3 vs 5, no profit (price decreased)
- Day 4: Price 6 vs 3, profit += 3 (6-3)
- Day 5: Price 4 vs 6, no profit (price decreased)
- Result: totalProfit = 7

### Time and Space Complexity

- **Time Complexity**: $O(n)$, where $n$ is the length of the prices array.
  Single pass through the array.
- **Space Complexity**: $O(1)$, using only a single variable.

### Test Cases

```typescript
console.log(maxProfitII([7, 1, 5, 3, 6, 4])); // Output: 7
console.log(maxProfitII([1, 2, 3, 4, 5])); // Output: 4 (profit on each consecutive day)
console.log(maxProfitII([7, 6, 4, 3, 1])); // Output: 0 (no profit possible)
```

## Conclusion

In this blog post, we've explored the **Best Time to Buy and Sell Stock**
problem, a staple in technical interviews that tests array traversal and greedy
algorithms. We started with an introduction to the problem, broke it down with
an ELI5 analogy (buying and selling candy for profit), and implemented a
solution in TypeScript with a time complexity of $O(n)$. We also tackled a bonus
problem, **Best Time to Buy and Sell Stock II**, which allows multiple
transactions and further reinforces the greedy approach.

Key takeaways:

- Use a greedy strategy to track the minimum price seen so far and update the
  maximum profit for the single-transaction version.
- For multiple transactions, capture every price increase as a separate profit
  opportunity.
- TypeScript's type annotations (like `number[]` for the input array) add
  clarity and prevent runtime errors.

These problems are excellent for practicing array manipulation and understanding
how greedy algorithms can simplify seemingly complex tasks. As you prepare for
Silicon Valley interviews, mastering such problems will boost your confidence in
handling real-world optimization challenges. Stay tuned for the next post in our
series, where we'll dive into more medium-difficulty LeetCode problems with
TypeScript!

import fs from "node:fs";
import path from "node:path";
import toml from "toml";
import { Feed } from "feed";
import { retext } from "retext";
import retextSmartypants from "retext-smartypants";
import type { BlogPost } from "./app/blog.js";

const blogDir = path.join("docs", "blog");
const filenames = fs
  .readdirSync(blogDir)
  .filter((filename) => filename.endsWith(".md"));

const blogPosts: BlogPost[] = filenames.map((filename) => {
  const filePath = path.join(blogDir, filename);
  const fileContent = fs.readFileSync(filePath, "utf8");

  const frontmatterDelimiter = "+++";
  const splitContent = fileContent.split(frontmatterDelimiter);
  const frontmatter = toml.parse(splitContent[1] as string);

  const title = retext()
    .use(retextSmartypants)
    .processSync(frontmatter.title as string)
    .toString();

  return {
    filename,
    title,
    author: frontmatter.author as string,
    date: frontmatter.date as string,
    content: "",
  };
});

const jsonPath = path.join("app", "blog.json");
fs.writeFileSync(jsonPath, JSON.stringify(blogPosts, null, 2));

const feed = new Feed({
  title: "Artintellica Blog",
  description: "Artintellica Blog",
  id: "https://artintellica.com/blog",
  link: "https://artintellica.com/blog",
  language: "en", // optional, used only in RSS 2.0, possible values: http://www.w3.org/TR/REC-html40/struct/dirlang.html#langcodes
  // image: "http://example.com/image.png",
  favicon: "https://artintellica.com/favicon.ico",
  copyright: `Copyright (C) ${new Date().getFullYear()} Ryan X. Charles`,
  updated: new Date(), // optional, default = today
  // generator: "awesome", // optional, default = 'Feed for Node.js'
  feedLinks: {
    json: "https://artintellica.com/blog/feed.json",
    atom: "https://example.com/blog/feed.atom.xml",
    rss: "https://example.com/blog/feed.rss.xml",
  },
  author: {
    name: "Ryan X. Charles",
    email: "artintellica@ryanxcharles.com",
    link: "https://artintellica.com",
  },
});

const recentPosts = blogPosts.reverse().slice(0, 20);
for (const post of recentPosts) {
  // date format: YYYY-MM-DD
  const stringDate = post.date;
  const date = new Date(
    Number.parseInt(stringDate.slice(0, 4)),
    Number.parseInt(stringDate.slice(5, 7)) - 1,
    Number.parseInt(stringDate.slice(8, 10)),
  );
  feed.addItem({
    title: post.title,
    id: `https://artintellica.com/blog/${post.filename}`,
    link: `https://artintellica.com/blog/${post.filename}`,
    date: date,
    description: post.title,
    content: post.content,
  });
}

const feedJsonPath = path.join("public", "blog", "feed.json");
fs.writeFileSync(feedJsonPath, feed.json1());
const feedAtomPath = path.join("public", "blog", "feed.atom.xml");
fs.writeFileSync(feedAtomPath, feed.atom1());
const feedRssPath = path.join("public", "blog", "feed.rss.xml");
fs.writeFileSync(feedRssPath, feed.rss2());

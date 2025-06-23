import fs from "node:fs";
import path from "node:path";
import toml from "toml";
import { retext } from "retext";
import retextSmartypants from "retext-smartypants";
import type { SeriesPost } from "./app/series.js";

const seriesDir = path.join("docs", "series");
const filenames = fs
  .readdirSync(seriesDir)
  .filter((filename) => filename.endsWith(".md"));

const seriesPosts: SeriesPost[] = filenames.map((filename) => {
  const filePath = path.join(seriesDir, filename);
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
    icon: frontmatter.icon as string,
    content: "",
  };
});

const jsonPath = path.join("app", "series.json");
fs.writeFileSync(jsonPath, JSON.stringify(seriesPosts, null, 2));

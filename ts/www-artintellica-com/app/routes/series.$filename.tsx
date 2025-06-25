import seriesPosts from "~app/series.json";
import { Link, href } from "react-router";
import MyMarkdown from "~app/components/my-markdown.js";
import fs from "fs";
import path from "path";
import type { Route } from "./+types/series.$filename.js";
import { $aicon } from "~app/util/aicons.js";
import { Footer } from "~app/components/footer.js";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
  const filename = params.filename;
  const newSeriesPosts = seriesPosts
    .map((post) => {
      return {
        title: post.title,
        date: post.date,
        author: post.author,
        filename: post.filename,
        content: "",
      };
    })
    .sort((a, b) => a.filename.localeCompare(b.filename))
    .reverse();

  const seriesPost = newSeriesPosts.find(
    (post) => post.filename === `${filename}`,
  );
  if (!seriesPost) {
    throw new Response(null, { status: 404, statusText: "Not found" });
  }
  // load content from app/series/filename
  const seriesDir = path.join("docs", "series");
  const filePath = path.join(seriesDir, `${filename}`);
  const contentRaw = fs.readFileSync(filePath, "utf8");
  const fmStart = contentRaw.indexOf("+++");
  if (fmStart === -1) {
    // No frontmatter, use whole file as content
    seriesPost.content = contentRaw;
  } else {
    const fmEnd = contentRaw.indexOf("+++", fmStart + 3);
    if (fmEnd === -1) {
      // Only one +++, invalid front-matter, maybe handle as error or raw content
      seriesPost.content = contentRaw;
    } else {
      // Content starts after the second +++
      seriesPost.content = contentRaw.slice(fmEnd + 3);
    }
  }

  // get five most recent series posts before this one
  const recentSeriesPosts = newSeriesPosts
    .filter((post) => post.filename.localeCompare(filename) < 0)
    .slice(0, 5);

  const nextSeriesPosts = newSeriesPosts
    .filter((post) => post.filename.localeCompare(filename) > 0)
    .reverse()
    .slice(0, 5);

  return { seriesPost, recentSeriesPosts, nextSeriesPosts };
};

export const meta = ({ data }: Route.MetaArgs) => {
  const seriesPostTitle = data?.seriesPost?.title || "Series Post";
  return [
    { title: `${seriesPostTitle} | Series | Artintellica` },
    { name: "description", content: "Open-source AI resources." },
  ];
};

export default function SeriesIndex({ loaderData }: Route.ComponentProps) {
  const { seriesPost, recentSeriesPosts, nextSeriesPosts } = loaderData;
  return (
    <div>
      <div className="mx-auto my-4 block aspect-square w-[120px]">
        <Link to={href("/")}>
          <img
            draggable={false}
            src={$aicon("/images/orange-cat-robot-300.webp")}
            alt=""
            className="block"
          />
        </Link>
      </div>
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div>
          <h1 className="my-4 text-center font-bold text-2xl text-black dark:text-white">
            {seriesPost.title}
          </h1>
          <div className="my-4 text-center text-black/60 text-sm dark:text-white/60">
            {seriesPost.date} &middot; {seriesPost.author}
          </div>
          <div className="text-black dark:text-white">
            <MyMarkdown>{seriesPost.content}</MyMarkdown>
          </div>
        </div>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      {nextSeriesPosts.length > 0 && (
        <div className="mx-auto my-4 max-w-[600px]">
          <h2 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Next Series Posts
          </h2>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {nextSeriesPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/series/:filename", { filename: post.filename })}
                    className="border-b border-b-blue font-semibold text-lg leading-3 hover:border-b-black dark:hover:border-b-white"
                  >
                    {post.title}
                  </Link>
                  <div className="text-black/60 text-sm dark:text-white/60">
                    {post.date} &middot; {post.author}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      {recentSeriesPosts.length > 0 && (
        <div className="mx-auto my-4 max-w-[600px]">
          <h2 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Earlier Series Posts
          </h2>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {recentSeriesPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/series/:filename", { filename: post.filename })}
                    className="border-b border-b-blue font-semibold text-lg leading-3 hover:border-b-black dark:hover:border-b-white"
                  >
                    {post.title}
                  </Link>
                  <div className="text-black/60 text-sm dark:text-white/60">
                    {post.date} &middot; {post.author}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="text-center text-black dark:text-white">
        <Link
          className="border-b border-b-blue font-bold text-lg hover:border-b-black dark:hover:border-b-white"
          to={href("/series")}
        >
          Back to Series
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <Footer />
    </div>
  );
}

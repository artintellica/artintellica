import blogPosts from "~app/blog.json";
import { Link, href } from "react-router";
import MyMarkdown from "~app/components/my-markdown.js";
import fs from "fs";
import path from "path";
import type { Route } from "./+types/blog.$filename.js";
import { $aicon } from "~app/util/aicons.js";
import { Footer } from "~app/components/footer.js";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
  const filename = params.filename;
  const newBlogPosts = blogPosts
    .map((post) => {
      return {
        title: post.title,
        date: post.date,
        author: post.author,
        filename: post.filename,
        code: post.code,
        content: "",
      };
    })
    .sort((a, b) => a.filename.localeCompare(b.filename))
    .reverse();

  const blogPost = newBlogPosts.find((post) => post.filename === `${filename}`);
  if (!blogPost) {
    throw new Response(null, { status: 404, statusText: "Not found" });
  }
  // load content from app/blog/filename
  const blogDir = path.join("docs", "blog");
  const filePath = path.join(blogDir, `${filename}`);
  const contentRaw = fs.readFileSync(filePath, "utf8");
  const fmStart = contentRaw.indexOf("+++");
  if (fmStart === -1) {
    // No frontmatter, use whole file as content
    blogPost.content = contentRaw;
  } else {
    const fmEnd = contentRaw.indexOf("+++", fmStart + 3);
    if (fmEnd === -1) {
      // Only one +++, invalid front-matter, maybe handle as error or raw content
      blogPost.content = contentRaw;
    } else {
      // Content starts after the second +++
      blogPost.content = contentRaw.slice(fmEnd + 3);
    }
  }

  // get five most recent blog posts before this one
  const recentBlogPosts = newBlogPosts
    .filter((post) => post.filename.localeCompare(filename) < 0)
    .slice(0, 5);

  const nextBlogPosts = newBlogPosts
    .filter((post) => post.filename.localeCompare(filename) > 0)
    .reverse()
    .slice(0, 5);

  return { blogPost, recentBlogPosts, nextBlogPosts };
};

export const meta = ({ data }: Route.MetaArgs) => {
  const blogPostTitle = data?.blogPost?.title || "Blog Post";
  return [
    { title: `${blogPostTitle} | Blog | Artintellica` },
    { name: "description", content: "Open-source AI resources." },
  ];
};

export default function BlogIndex({ loaderData }: Route.ComponentProps) {
  const { blogPost, recentBlogPosts, nextBlogPosts } = loaderData;
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
            {blogPost.title}
          </h1>
          <div className="my-4 text-center text-black/60 text-sm dark:text-white/60">
            {blogPost.date} &middot; {blogPost.author}
          </div>
          {blogPost.code && (
            <div className="my-4 text-center text-black/60 text-sm dark:text-white/60">
              <Link
                to={blogPost.code}
                className="border-b border-b-blue font-semibold text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
              >
                View Source Code on GitHub
              </Link>
            </div>
          )}
          <div className="text-black dark:text-white">
            <MyMarkdown>{blogPost.content}</MyMarkdown>
          </div>
        </div>
      </div>
      {blogPost.code && (
        <div>
          <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
          <div className="my-4 text-center text-black/60 text-sm dark:text-white/60">
            <Link
              to={blogPost.code}
              className="border-b border-b-blue font-semibold text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
            >
              View Source Code on GitHub
            </Link>
          </div>
        </div>
      )}
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      {nextBlogPosts.length > 0 && (
        <div className="mx-auto my-4 max-w-[600px]">
          <h2 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Next Blog Posts
          </h2>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {nextBlogPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/blog/:filename", { filename: post.filename })}
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
      {recentBlogPosts.length > 0 && (
        <div className="mx-auto my-4 max-w-[600px]">
          <h2 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Earlier Blog Posts
          </h2>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {recentBlogPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={href("/blog/:filename", { filename: post.filename })}
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
          to={href("/blog")}
        >
          Back to Blog
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <Footer />
    </div>
  );
}

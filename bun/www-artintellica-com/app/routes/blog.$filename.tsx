import blogPosts from "~app/blog.json";
import { Link } from "react-router";
import MyMarkdown from "~app/components/my-markdown.js";
import { $path } from "safe-routes";
import fs from "fs";
import path from "path";
import type { Route } from "./+types/blog.$filename.js";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
  const filename = params.filename;
  const newBlogPosts = blogPosts
    .map((post) => {
      return {
        title: post.title,
        date: post.date,
        author: post.author,
        filename: post.filename,
        content: "",
      };
    })
    .sort((a, b) => a.date.localeCompare(b.date))
    .reverse();

  const blogPost = newBlogPosts.find((post) => post.filename === `${filename}`);
  if (!blogPost) {
    throw new Response(null, { status: 404, statusText: "Not found" });
  }
  // load content from app/blog/filename
  const blogDir = path.join("docs", "blog");
  const filePath = path.join(blogDir, `${filename}`);
  const fileContent = fs
    .readFileSync(filePath, "utf8")
    .split("+++")[2] as string;
  blogPost.content = fileContent;

  // get five most recent blog posts before this one
  const recentBlogPosts = newBlogPosts
    .filter((post) => post.date < blogPost.date)
    .slice(0, 5);

  return { blogPost, recentBlogPosts };
};

export const meta: Route.MetaFunction = ({ data }) => {
  const blogPost = data.blogPost;
  return [
    { title: `${blogPost.title} | Blog | Artintellica` },
    { name: "description", content: "Welcome to Artintellica!" },
  ];
};

export default function BlogIndex({ loaderData }: Route.ComponentProps) {
  const { blogPost, recentBlogPosts } = loaderData;
  return (
    <div>
      <div className="mx-auto my-4">
        <Link to={$path("/")}>
          <Logo />
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
          <div className="text-black dark:text-white">
            <MyMarkdown>{blogPost.content}</MyMarkdown>
          </div>
        </div>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="mx-auto my-4 max-w-[600px]">
        <div>
          <h2 className="my-4 text-center font-bold text-black text-xl dark:text-white">
            Earlier Blog Posts
          </h2>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {recentBlogPosts.map((post) => (
                <div key={post.filename} className="mb-4">
                  <Link
                    to={$path("/blog/:filename", { filename: post.filename })}
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
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="text-center text-black dark:text-white">
        <Link
          className="border-b border-b-blue font-bold text-lg hover:border-b-black dark:hover:border-b-white"
          to={$path("/blog")}
        >
          Back to Blog
        </Link>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="mx-auto mb-4">
        <div className="mb-4 text-center text-black text-sm dark:text-white" />
        <div className="text-center text-black/70 text-sm dark:text-white/70">
          Copyright &copy; {new Date().getFullYear()} Identellica LLC
          <br />
          <Link
            to={$path("/")}
            className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
          >
            Home
          </Link>
          <span> &middot; </span>
          <Link
            to="/blog"
            className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
          >
            Blog
          </Link>
        </div>
      </div>
    </div>
  );
}

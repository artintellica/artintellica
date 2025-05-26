import blogPosts from "~app/blog.json";
import { Link } from "react-router";
import { $path } from "safe-routes";
import type { Route } from "./+types/blog._index.js";
import { $aicon } from "~app/util/aicons.js";

export const loader = async ({ request, params }: Route.LoaderArgs) => {
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
  return { blogPosts: newBlogPosts };
};

export const meta: Route.MetaFunction = () => {
  return [
    { title: "Blog | Artintellica" },
    { name: "description", content: "Welcome to Artintellica!" },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/json",
      title: "JSON Feed",
      href: "/blog/feed.json",
    },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/atom+xml",
      title: "Atom Feed",
      href: "/blog/feed.atom.xml",
    },
    {
      tagName: "link",
      rel: "alternate",
      type: "application/rss+xml",
      title: "RSS Feed",
      href: "/blog/feed.rss.xml",
    },
  ];
};

export default function BlogIndex({ loaderData }: Route.ComponentProps) {
  const { blogPosts } = loaderData;
  return (
    <div>
      <div className="mx-auto my-4 block aspect-square w-[120px]">
        <img
          draggable={false}
          src={$aicon("/images/orange-cat-robot-300.webp")}
          alt=""
          className="block"
        />
      </div>
      <div className="mx-auto my-4 max-w-[600px] px-2">
        <div>
          <h1 className="my-4 text-center font-bold text-2xl text-black dark:text-white">
            Blog
          </h1>
          <div>
            <div className="mb-4 text-black dark:text-white">
              {blogPosts.map((post) => (
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
      <div className="text-center text-black text-sm dark:text-white">
        <h2 className="font-bold text-lg">Blog Feeds</h2>
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/blog/feed.json"}
        >
          JSON Feed
        </a>
        &nbsp;&middot;&nbsp;
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/blog/feed.atom.xml"}
        >
          Atom Feed
        </a>
        &nbsp;&middot;&nbsp;
        <a
          className="border-b border-b-blue text-sm hover:border-b-black dark:hover:border-b-white"
          href={"/blog/feed.rss.xml"}
        >
          RSS Feed
        </a>
      </div>
      <hr className="mx-auto my-4 max-w-[40px] border-black/40 dark:border-white/40" />
      <div className="text-center text-black dark:text-white">
        <Link
          className="border-b border-b-blue font-bold text-lg hover:border-b-black dark:hover:border-b-white"
          to={$path("/")}
        >
          Back to Home
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

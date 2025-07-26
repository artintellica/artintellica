import { Link, href } from "react-router";

export function Footer() {
  return (
    <div className="mx-auto mb-4">
      <div className="mb-4 text-center text-black text-sm dark:text-white" />
      <div className="text-center text-black/70 text-sm dark:text-white/70">
        Copyright &copy; {new Date().getFullYear()}{" "}
        <Link
          to="https://identellica.com"
          className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
        >
          EarthBucks Inc.
        </Link>
        <br />
        <Link
          to={href("/")}
          className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
        >
          Home
        </Link>
        <span> &middot; </span>
        <Link
          to={href("/blog")}
          className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
        >
          Blog
        </Link>
        <span> &middot; </span>
        <Link
          to="https://github.com/artintellica/artintellica"
          className="border-b border-b-blue text-black hover:border-b-black dark:text-white dark:hover:border-b-white"
        >
          Source Code
        </Link>
      </div>
    </div>
  );
}

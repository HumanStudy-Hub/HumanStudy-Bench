import Link from "next/link";

const navLinks = [
  { name: "Overview", href: "/#overview", icon: "overview" },
  { name: "Leaderboard", href: "/#leaderboard", icon: "leaderboard" },
  { name: "Dataset", href: "/#dataset", icon: "dataset" },
  { name: "Paper", href: "https://arxiv.org/abs/2602.00685", icon: "paper" },
  { name: "GitHub", href: "https://github.com/HumanStudy-Hub/HumanStudy-Bench", icon: "github" },
];

const actionLinks = [
  { name: "Contribute Now", href: "/contribute", button: true },
];

export default function Navbar() {
  return (
    <nav className="border-b border-gray-200 bg-white sticky top-0 z-50 bg-opacity-90 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between items-center">
          <div className="flex">
            <Link href="/" className="flex flex-shrink-0 items-center font-bold text-xl text-black no-underline hover:no-underline">
               <span className="font-serif tracking-tight">HumanStudy-Bench</span>
            </Link>
          </div>
          
          <div className="hidden sm:ml-6 sm:flex sm:items-center sm:space-x-4">
            {/* Primary Nav */}
            <div className="flex space-x-6 mr-4 border-r border-gray-100 pr-4">
              {navLinks.map((link) => {
                const isExternal = link.href.startsWith("http");
                const className =
                  "inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-500 hover:text-cyan-600 transition-colors";
                const content = (
                  <>
                  {link.icon === "overview" && (
                    <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
                  )}
                  {link.icon === "leaderboard" && (
                    <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3v18h18M7 15h2v3H7v-3zm4-6h2v9h-2V9zm4 3h2v6h-2v-6z" /></svg>
                  )}
                  {link.icon === "dataset" && (
                    <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3" strokeWidth={2} /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5v6c0 1.657 3.134 3 7 3s7-1.343 7-3V5M5 11v6c0 1.657 3.134 3 7 3s7-1.343 7-3v-6" /></svg>
                  )}
                  {link.icon === "paper" && (
                    <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
                  )}
                  {link.icon === "github" && (
                    <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 24 24"><path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" /></svg>
                  )}
                  {link.name}
                  </>
                );
                if (isExternal) {
                  return (
                    <a
                      key={link.name}
                      href={link.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={className}
                    >
                      {content}
                    </a>
                  );
                }
                return (
                  <Link key={link.name} href={link.href} className={className}>
                    {content}
                  </Link>
                );
              })}
            </div>

            {/* Boxed Actions */}
            <div className="flex items-center bg-gray-50 border border-gray-200 rounded-lg p-1.5 space-x-2 shadow-sm">
              {actionLinks.map((link) => (
                <Link
                  key={link.name}
                  href={link.href}
                  className={link.button 
                    ? "inline-flex items-center rounded-md bg-cyan-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-cyan-500 transition-colors whitespace-nowrap"
                    : "inline-flex items-center px-3 py-1.5 text-xs font-semibold text-gray-700 hover:text-cyan-600 transition-colors whitespace-nowrap"
                  }
                >
                  {link.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

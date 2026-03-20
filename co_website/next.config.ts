/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  async redirects() {
    return [
      // Serve our custom icon when browser requests favicon.ico (avoids Vercel default)
      { source: "/favicon.ico", destination: "/favicon.svg", permanent: false },
    ];
  },
};

module.exports = nextConfig;
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [],
    unoptimized: true,
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
        ],
      },
    ];
  },
  // API proxy is implemented in app/api/proxy/[...path]/route.ts so the backend URL
  // is read at request time from env (fixes wrong host baked in at `next build`).
};

export default nextConfig;

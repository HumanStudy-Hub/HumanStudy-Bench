export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <p className="text-center text-sm text-gray-500">
          &copy; {new Date().getFullYear()} Xuan Liu, Haoyang Shang, Zizhang Liu, Guankai Zhai. All rights reserved.
        </p>
      </div>
    </footer>
  );
}

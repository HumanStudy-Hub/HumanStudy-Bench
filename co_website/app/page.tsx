import Hero from "@/components/Hero";
import Overview from "@/components/Overview";
import DatasetGrid from "@/components/DatasetGrid";
import LeaderboardTable from "@/components/LeaderboardTable";

export default function Home() {
  return (
    <div className="bg-white">
       <Hero />
      <Overview />
      <LeaderboardTable />
      <DatasetGrid />
    </div>
  );
}

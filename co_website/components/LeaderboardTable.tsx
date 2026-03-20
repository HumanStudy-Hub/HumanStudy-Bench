import { promises as fs } from 'fs';
import path from 'path';
import LeaderboardList from './LeaderboardList';

export default async function LeaderboardTable() {
  const filePath = path.join(process.cwd(), 'data/leaderboard.json');
  let data = [];
  try {
      const fileContents = await fs.readFile(filePath, 'utf8');
      data = JSON.parse(fileContents);
  } catch (e) {
      console.error("Could not read leaderboard data", e);
  }

  return (
    <div id="leaderboard" className="py-24 bg-gray-50 sm:py-32 border-t border-gray-200">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <div className="mx-auto max-w-2xl text-center mb-10">
                <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">Leaderboard</h2>
                <p className="mt-4 text-lg leading-8 text-gray-600">
                    Evaluating agent design alignment with human behavior using Probability Alignment Score (PAS) and Effect Consistency Score (ECS) across 12 foundational human-subject studies.
                </p>
            </div>

            <LeaderboardList rawData={data} />
        </div>
    </div>
  );
}

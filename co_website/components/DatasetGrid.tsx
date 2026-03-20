const datasets = [
  {
    title: "The False Consensus Effect",
    authors: "Ross et al., 1977",
    tags: ["Individual Cognition"],
    phenomenon: "False consensus bias",
    color: "bg-cyan-50 text-cyan-700 border-cyan-200"
  },
  {
    title: "Measures of Anchoring",
    authors: "Jacowitz & Kahneman, 1995",
    tags: ["Individual Cognition"],
    phenomenon: "Anchoring effect",
     color: "bg-cyan-50 text-cyan-700 border-cyan-200"
  },
  {
    title: "Framing of Decisions",
    authors: "Tversky & Kahneman, 1981",
    tags: ["Individual Cognition"],
    phenomenon: "Framing effect",
     color: "bg-cyan-50 text-cyan-700 border-cyan-200"
  },
  {
    title: "Subjective Probability",
    authors: "Kahneman & Tversky, 1972",
    tags: ["Individual Cognition"],
    phenomenon: "Representativeness heuristic",
     color: "bg-cyan-50 text-cyan-700 border-cyan-200"
  },
  {
    title: "Intentional Action",
    authors: "Knobe, 2003",
    tags: ["Social Psychology"],
    phenomenon: "Knobe effect",
    color: "bg-sky-50 text-sky-700 border-sky-200"
  },
  {
    title: "Forming Impressions",
    authors: "Asch, 1946",
    tags: ["Social Psychology"],
    phenomenon: "Primacy effect",
    color: "bg-sky-50 text-sky-700 border-sky-200"
  },
  {
    title: "Social Categorization",
    authors: "Billig & Tajfel, 1973",
    tags: ["Social Psychology"],
    phenomenon: "Minimal group paradigm",
     color: "bg-sky-50 text-sky-700 border-sky-200"
  },
  {
    title: "Pluralistic Ignorance",
    authors: "Prentice & Miller, 1993",
    tags: ["Social Psychology"],
    phenomenon: "Pluralistic ignorance",
     color: "bg-sky-50 text-sky-700 border-sky-200"
  },
  {
    title: "Guessing Games",
    authors: "Nagel, 1995",
    tags: ["Strategic Interaction"],
    phenomenon: "Keynesian beauty contest",
     color: "bg-blue-50 text-blue-700 border-blue-200"
  },
  {
    title: "Thinking through Uncertainty",
    authors: "Shafir & Tversky, 1992",
    tags: ["Strategic Interaction"],
    phenomenon: "Disjunction effect",
     color: "bg-blue-50 text-blue-700 border-blue-200"
  },
  {
    title: "Fairness in Bargaining",
    authors: "Forsythe et al., 1994",
    tags: ["Strategic Interaction"],
    phenomenon: "Dictator game giving",
     color: "bg-blue-50 text-blue-700 border-blue-200"
  },
  {
    title: "Trust and Reciprocity",
    authors: "Berg et al., 1995",
    tags: ["Strategic Interaction"],
    phenomenon: "Trust game",
     color: "bg-blue-50 text-blue-700 border-blue-200"
  }
];

export default function DatasetGrid() {
  return (
    <div id="dataset" className="py-24 bg-white sm:py-32 border-t border-gray-100">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">Study Dataset</h2>
          <p className="mt-4 text-lg leading-8 text-gray-600">
            An initial curated collection of 12 foundational human-subject studies spanning individual cognition, strategic interaction, and social psychology, all with complete experimental materials and clearly specified statistical tests.
          </p>
          <a href="/contribute" className="mt-3 inline-block text-lg font-medium text-indigo-600 hover:text-indigo-500">
            Become a contributor &rarr;
          </a>
        </div>
        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:mt-20 lg:max-w-none lg:grid-cols-2">
          {datasets.map((study) => (
            <div key={study.title} className="flex flex-col gap-y-4 rounded-2xl border border-gray-200 p-6 hover:shadow-md transition-shadow">
               <div className="flex items-center justify-between">
                 <h3 className="text-lg font-semibold leading-8 text-gray-900 font-serif">
                   {study.title}
                 </h3>
                 <span className="text-sm text-gray-500 italic">{study.authors}</span>
               </div>
               
               <div className="flex flex-wrap gap-2">
                  {study.tags.map(tag => (
                      <span key={tag} className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium ${study.color}`}>
                        {tag}
                      </span>
                  ))}
               </div>
               
               <div className="mt-2 text-sm text-gray-600">
                 <strong className="text-gray-900">Phenomenon:</strong> {study.phenomenon}
               </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

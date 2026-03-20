export default function Overview() {
  return (
    <div id="overview" className="bg-white py-24 sm:py-32">
       {/* Introduction Section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 mb-16">
        <div className="mx-auto max-w-4xl">
          <div className="text-base leading-7 text-gray-700 text-center">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">
              What is HumanStudy-Bench?
            </h1>
            <div className="max-w-3xl mx-auto">
              <p className="mt-6 text-lg">
              HumanStudy-Bench treats participant simulation as an <strong>agent design problem</strong> and provides a standardized testbed — combining an <strong>Execution Engine</strong> that reconstructs full experimental protocols from published studies and a <strong>Benchmark</strong> with standardized evaluation metrics — for <em>replaying human-subject experiments end-to-end</em> with alignment evaluation at the level of scientific inference.
              </p>
              <div className="mt-8 bg-cyan-50 py-5 px-6 rounded-lg shadow-sm">
                  <h3 className="font-bold text-gray-900 font-serif text-lg text-center">Standardized Testbed</h3>
                  <p className="text-sm text-gray-700 mt-2 text-center">Test different agent designs on the same experiments, run agents through real studies covering 6,000+ trials, and compare results rigorously using inferential-level metrics.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Stats Section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 mb-16">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          <div className="bg-gradient-to-br from-cyan-50 to-sky-50 rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-cyan-500 mb-2">12</div>
            <div className="text-sm font-semibold text-gray-900">Foundational Studies</div>
            <div className="text-xs text-gray-600 mt-1">Covering major behavioral phenomena</div>
          </div>
          <div className="bg-gradient-to-br from-sky-50 to-blue-50 rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-sky-500 mb-2">6,000+</div>
            <div className="text-sm font-semibold text-gray-900">Experimental Trials</div>
            <div className="text-xs text-gray-600 mt-1">Replayed with AI agents</div>
          </div>
          <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-blue-500 mb-2">10-2,000+</div>
            <div className="text-sm font-semibold text-gray-900">Human Sample Range</div>
            <div className="text-xs text-gray-600 mt-1">Per study participant count</div>
          </div>
          <div className="bg-gradient-to-br from-sky-50 to-cyan-50 rounded-xl p-6 text-center">
            <div className="text-3xl font-bold text-sky-500 mb-2">2</div>
            <div className="text-sm font-semibold text-gray-900">Evaluation Metrics</div>
            <div className="text-xs text-gray-600 mt-1">PAS & ECS for alignment</div>
          </div>
        </div>
        <div className="mt-6 text-center">
          <a href="/contribute" className="inline-flex items-center gap-1.5 text-sm font-medium text-indigo-600 hover:text-indigo-500 transition-colors">
            Open-ended and growing — join us to add more
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
            </svg>
          </a>
        </div>
      </div>

       {/* Visual Pipeline Section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl font-serif">Pipeline Architecture</h2>
          <p className="mt-4 text-lg leading-8 text-gray-600">
            From published human studies to reusable simulation environment in four stages.
          </p>
        </div>
        <div className="mx-auto mt-8 max-w-2xl sm:mt-12 lg:mt-16 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-6 gap-y-8 lg:max-w-none lg:grid-cols-4">
            
            {/* Stage 1: Filter */}
            <div className="flex flex-col bg-gradient-to-br from-cyan-50 to-white p-6 rounded-2xl shadow-md hover:shadow-lg transition-shadow relative z-10">
              <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 mb-2">
                 <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-cyan-500 shadow-md">
                    <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                    </svg>
                 </div>
                 <span className="text-lg">Stage 1: Filter</span>
              </dt>
              <dd className="flex flex-auto flex-col text-sm leading-6 text-gray-700">
                <p className="flex-auto">Curates human studies that are scientifically important and practically reproducible, ensuring full experimental details, quantifiable outcomes, and simulation feasibility.</p>
              </dd>
            </div>

            {/* Stage 2: Extract */}
            <div className="flex flex-col bg-gradient-to-br from-sky-50 to-white p-6 rounded-2xl shadow-md hover:shadow-lg transition-shadow relative z-10">
              <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 mb-2">
                 <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-sky-500 shadow-md">
                    <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                 </div>
                 <span className="text-lg">Stage 2: Extract</span>
              </dt>
              <dd className="flex flex-auto flex-col text-sm leading-6 text-gray-700">
                <p className="flex-auto">Extracts participants&apos; profiles, experimental designs, statistical tests, and human ground-truth outcomes from unstructured papers into machine-executable representations.</p>
              </dd>
            </div>

            {/* Stage 3: Execute */}
            <div className="flex flex-col bg-gradient-to-br from-blue-50 to-white p-6 rounded-2xl shadow-md hover:shadow-lg transition-shadow relative z-10">
               <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 mb-2">
                 <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-blue-500 shadow-md">
                    <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                 </div>
                 <span className="text-lg">Stage 3: Execute</span>
              </dt>
              <dd className="flex flex-auto flex-col text-sm leading-6 text-gray-700">
                <p className="flex-auto">Runs agent designs through reconstructed experimental protocols, generating trial-level data via a shared execution engine that handles agent sampling, instruction dispatch, and response collection.</p>
              </dd>
            </div>

            {/* Stage 4: Evaluate */}
            <div className="flex flex-col bg-gradient-to-br from-blue-50 to-white p-6 rounded-2xl shadow-md hover:shadow-lg transition-shadow relative z-10">
               <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 mb-2">
                 <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-blue-600 shadow-md">
                    <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                 </div>
                 <span className="text-lg">Stage 4: Evaluate</span>
              </dt>
              <dd className="flex flex-auto flex-col text-sm leading-6 text-gray-700">
                <p className="flex-auto">Compares agent responses against human ground-truth using Probability Alignment Score (PAS) for inferential agreement and Effect Consistency Score (ECS) for effect-size alignment.</p>
              </dd>
            </div>

          </dl>
          
          <div className="mt-12 text-left border-t-2 border-gray-300 pt-10">
              <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center font-serif">Evaluation Metrics</h3>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                 <div className="rounded-xl bg-gradient-to-br from-blue-50 to-sky-50 p-6 shadow-md hover:shadow-lg transition-shadow">
                    <h3 className="flex items-center font-bold text-gray-900 text-lg mb-3">
                        <span className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500 text-white text-base font-bold mr-3 shadow-md">PAS</span>
                        Probability Alignment Score
                    </h3>
                    <p className="text-gray-700 text-sm leading-relaxed">Measures whether agents reach the same scientific conclusions as humans at the phenomenon level. It quantifies the probability that agent and human populations exhibit behavior consistent with the same hypothesis.</p>
                 </div>
                 <div className="rounded-xl bg-gradient-to-br from-sky-50 to-cyan-50 p-6 shadow-md hover:shadow-lg transition-shadow">
                    <h3 className="flex items-center font-bold text-gray-900 text-lg mb-3">
                        <span className="flex h-10 w-10 items-center justify-center rounded-full bg-sky-500 text-white text-base font-bold mr-3 shadow-md">ECS</span>
                        Effect Consistency Score
                    </h3>
                    <p className="text-gray-700 text-sm leading-relaxed">Measures how closely agents reproduce the magnitude and pattern of human behavioral effects at the data level. It assesses both the precision and accuracy of agent responses compared to human ground truth.</p>
                 </div>
              </div>
          </div>

        </div>
      </div>
    </div>
  );
}

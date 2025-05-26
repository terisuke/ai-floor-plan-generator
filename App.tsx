
import React, { useState, useCallback } from 'react';
import { GridDimensions } from './types';
import GridInputForm from './components/GridInputForm';
import ImageDisplay from './components/ImageDisplay';
import { generateFloorPlanImage } from './services/floorPlanService';

const App: React.FC = () => {
  const [gridDimensions, setGridDimensions] = useState<GridDimensions | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateFloorPlan = useCallback(async (dimensions: GridDimensions) => {
    setIsLoading(true);
    setError(null);
    setGeneratedImage(null); // Clear previous image
    setGridDimensions(dimensions);

    try {
      // Simulate a short delay before calling the service for better UX
      await new Promise(resolve => setTimeout(resolve, 300));
      const imageUrl = await generateFloorPlanImage(dimensions);
      setGeneratedImage(imageUrl);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred during image generation.');
      }
      setGeneratedImage(null); // Ensure no broken image link is shown
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100 flex flex-col items-center justify-center p-4 sm:p-6 font-sans">
      <div className="w-full max-w-3xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 pb-2">
            AI Floor Plan Studio
          </h1>
          <p className="text-slate-400 mt-2 text-md sm:text-lg">
            Define your grid, and let our (simulated) AI craft a conceptual floor plan.
          </p>
        </header>

        <main className="bg-slate-800/70 backdrop-blur-md shadow-2xl rounded-xl p-6 sm:p-10 border border-slate-700">
          <GridInputForm onSubmit={handleGenerateFloorPlan} isLoading={isLoading} />
          
          {error && (
            <div className="mt-6 p-4 bg-red-700/30 text-red-300 border border-red-600 rounded-lg text-sm sm:text-base flex items-center space-x-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span><strong>Error:</strong> {error}</span>
            </div>
          )}

          <ImageDisplay 
            imageUrl={generatedImage} 
            isLoading={isLoading} 
            dimensions={gridDimensions} 
          />
        </main>

        <footer className="text-center mt-10 text-slate-500 text-xs sm:text-sm">
          <p>&copy; {new Date().getFullYear()} FloorPlan AI MVP. For demonstration purposes only.</p>
          <p>Images are placeholders from picsum.photos.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;


import React from 'react';
import { GridDimensions } from '../types';
import LoadingSpinner from './LoadingSpinner';

interface ImageDisplayProps {
  imageUrl: string | null;
  isLoading: boolean;
  dimensions: GridDimensions | null;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ imageUrl, isLoading, dimensions }) => {
  if (isLoading) {
    return (
      <div className="mt-8 p-6 bg-slate-700/50 rounded-lg h-80 flex flex-col items-center justify-center text-slate-300 border border-slate-600">
        <LoadingSpinner />
        <p className="mt-4 text-lg">Generating your floor plan...</p>
        {dimensions && <p className="text-sm text-slate-400">Dimensions: {dimensions.width} x {dimensions.height}</p>}
      </div>
    );
  }

  if (!imageUrl && !dimensions) { // Initial state, before any generation attempt
    return (
      <div className="mt-8 p-6 bg-slate-700/50 rounded-lg h-80 flex flex-col items-center justify-center text-slate-400 border border-slate-600 border-dashed">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-slate-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <p className="text-lg">Your generated floor plan will appear here.</p>
        <p className="text-sm">Enter grid dimensions above and click "Generate".</p>
      </div>
    );
  }
  
  if (!imageUrl && dimensions) { // Error state or no image after generation attempt
     return (
      <div className="mt-8 p-6 bg-slate-700/50 rounded-lg h-80 flex flex-col items-center justify-center text-slate-400 border border-slate-600">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-yellow-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-lg">Could not display image.</p>
        {dimensions && <p className="text-sm text-slate-500">Requested dimensions: {dimensions.width} x {dimensions.height}</p>}
      </div>
    );
  }


  return (
    <div className="mt-8 p-4 bg-slate-700/50 rounded-lg border border-slate-600 shadow-lg">
      {dimensions && (
        <p className="text-sm text-slate-400 mb-3 text-center">
          Displaying conceptual floor plan for grid: <span className="font-semibold text-slate-200">{dimensions.width} x {dimensions.height}</span>
        </p>
      )}
      <div className="aspect-[4/3] bg-slate-600 rounded-md overflow-hidden flex items-center justify-center">
        <img 
          src={imageUrl!} 
          alt={`Generated Floor Plan ${dimensions ? `(${dimensions.width}x${dimensions.height})` : ''}`} 
          className="w-full h-full object-contain"
          onError={(e) => {
            // This is a fallback if the image URL itself is broken, though the service should handle errors.
            (e.target as HTMLImageElement).style.display = 'none'; 
            // Optionally, replace with a placeholder or message directly in the img container
            const parent = (e.target as HTMLImageElement).parentElement;
            if(parent){
              parent.innerHTML = '<p class="text-red-400 p-4">Error loading image.</p>';
            }
          }}
        />
      </div>
    </div>
  );
};

export default ImageDisplay;


import React, { useState } from 'react';
import { GridDimensions } from '../types';

interface GridInputFormProps {
  onSubmit: (dimensions: GridDimensions) => void;
  isLoading: boolean;
}

const GridInputForm: React.FC<GridInputFormProps> = ({ onSubmit, isLoading }) => {
  const [width, setWidth] = useState<string>('10');
  const [height, setHeight] = useState<string>('11');
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setValidationError(null);

    const numWidth = parseInt(width, 10);
    const numHeight = parseInt(height, 10);

    if (isNaN(numWidth) || isNaN(numHeight) || numWidth <= 0 || numHeight <= 0) {
      setValidationError('Please enter valid positive numbers for width and height.');
      return;
    }
    if (numWidth > 50 || numHeight > 50) {
        setValidationError('Dimensions are too large. Max 50x50 for this demo.');
        return;
    }

    onSubmit({ width: numWidth, height: numHeight });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div>
          <label htmlFor="gridWidth" className="block text-sm font-medium text-slate-300 mb-1">
            Grid Width (e.g., 10)
          </label>
          <input
            type="number"
            id="gridWidth"
            value={width}
            onChange={(e) => setWidth(e.target.value)}
            placeholder="e.g., 10"
            className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-colors duration-150"
            min="1"
            max="50" // Example constraint
            disabled={isLoading}
          />
        </div>
        <div>
          <label htmlFor="gridHeight" className="block text-sm font-medium text-slate-300 mb-1">
            Grid Height (e.g., 11)
          </label>
          <input
            type="number"
            id="gridHeight"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            placeholder="e.g., 11"
            className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-colors duration-150"
            min="1"
            max="50" // Example constraint
            disabled={isLoading}
          />
        </div>
      </div>

      {validationError && (
        <p className="text-sm text-red-400 bg-red-700/30 border border-red-600 p-3 rounded-md">{validationError}</p>
      )}

      <button
        type="submit"
        disabled={isLoading}
        className="w-full px-6 py-3.5 text-base font-semibold text-white bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg shadow-md hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:ring-offset-2 focus:ring-offset-slate-800 transition-all duration-150 ease-in-out disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center"
      >
        {isLoading ? (
          <>
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Generating...
          </>
        ) : (
          'Generate Floor Plan'
        )}
      </button>
    </form>
  );
};

export default GridInputForm;


import { GridDimensions, ApiError } from '../types';

const API_BASE_URL = 'http://localhost:3001';

export const generateFloorPlanImage = async (dimensions: GridDimensions): Promise<string> => {
  console.log('Calling backend API to generate floor plan with dimensions:', dimensions);

  if (dimensions.width <=0 || dimensions.height <=0) {
    throw new Error('Invalid dimensions: Width and height must be positive.');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/generate-floor-plan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        width: dimensions.width,
        height: dimensions.height,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const error: ApiError = { 
        message: errorData.error || `API error: ${response.status} ${response.statusText}` 
      };
      console.error('API Error:', error);
      throw new Error(error.message);
    }

    const data = await response.json();
    const imageUrl = `${API_BASE_URL}${data.imageUrl}`;
    
    console.log('API success. Generated image URL:', imageUrl);
    return imageUrl;
  } catch (error) {
    console.error('Failed to generate floor plan:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to connect to the backend API. Please ensure the backend server is running.');
  }
};

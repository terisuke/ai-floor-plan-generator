import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());
app.use('/generated', express.static(path.join(__dirname, '../../generated')));

interface GenerateRequest {
  width: number;
  height: number;
}

app.post('/api/generate-floor-plan', async (req, res) => {
  try {
    const { width, height }: GenerateRequest = req.body;

    if (!width || !height || width < 1 || height < 1 || width > 50 || height > 50) {
      return res.status(400).json({ error: 'Invalid dimensions. Width and height must be between 1 and 50.' });
    }

    console.log(`Generating floor plan with dimensions: ${width}x${height}`);

    // Using LoRA-finetuned model for improved floor plan generation
    const pythonScriptPath = path.join(__dirname, '../../inference/generate_floor_plan_lora.py');
    const outputDir = path.join(__dirname, '../../generated');
    
    await fs.mkdir(outputDir, { recursive: true });

    const timestamp = Date.now();
    const outputFilename = `floor_plan_${width}x${height}_${timestamp}.png`;
    const outputPath = path.join(outputDir, outputFilename);

    const pythonProcess = spawn('python3', [
      pythonScriptPath,
      '--width', width.toString(),
      '--height', height.toString(),
      '--output', outputPath
    ]);

    let errorData = '';

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        console.error(`Python script exited with code ${code}`);
        return res.status(500).json({ error: `Failed to generate floor plan: ${errorData}` });
      }

      try {
        await fs.access(outputPath);
        const imageUrl = `/generated/${outputFilename}`;
        res.json({ imageUrl });
      } catch (error) {
        console.error('Generated file not found:', error);
        res.status(500).json({ error: 'Failed to generate floor plan' });
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      res.status(500).json({ error: 'Failed to start generation process' });
    });

  } catch (error) {
    console.error('Error in generate-floor-plan endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});

import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { RotateCcw, Thermometer, Hash, Target, Layers } from 'lucide-react';
import { motion } from 'framer-motion';

interface ModelConfig {
  temperature: number;
  maxTokens: number;
  topK: number;
  topP: number;
  preset: 'creative' | 'balanced' | 'precise';
}

interface SettingsPanelProps {
  config: ModelConfig;
  onConfigChange: (config: ModelConfig) => void;
}

const presets = {
  creative: { temperature: 1.2, maxTokens: 200, topK: 80, topP: 0.95 },
  balanced: { temperature: 0.7, maxTokens: 150, topK: 50, topP: 0.9 },
  precise: { temperature: 0.3, maxTokens: 100, topK: 20, topP: 0.8 },
};

export const SettingsPanel = ({ config, onConfigChange }: SettingsPanelProps) => {
  const handlePresetChange = (preset: keyof typeof presets) => {
    onConfigChange({
      ...config,
      ...presets[preset],
      preset,
    });
  };

  const handleReset = () => {
    onConfigChange({
      temperature: 0.7,
      maxTokens: 150,
      topK: 50,
      topP: 0.9,
      preset: 'balanced',
    });
  };

  const updateConfig = (key: keyof ModelConfig, value: number) => {
    onConfigChange({
      ...config,
      [key]: value,
    });
  };

  return (
    <div className="h-full overflow-y-auto p-4 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Layers className="h-5 w-5 text-blue-400" />
          <h2 className="text-lg font-semibold text-white">Model Configuration</h2>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          className="text-slate-400 hover:text-white"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      {/* Presets */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-3">Presets</h3>
        <div className="grid grid-cols-1 gap-2">
          {Object.entries(presets).map(([key, preset]) => (
            <motion.button
              key={key}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handlePresetChange(key as keyof typeof presets)}
              className={`p-3 rounded-lg border text-left transition-all ${
                config.preset === key
                  ? 'border-blue-400 bg-blue-400/10 text-blue-300'
                  : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium capitalize">{key}</span>
                {config.preset === key && (
                  <Badge variant="secondary" className="bg-blue-400/20 text-blue-300">
                    Active
                  </Badge>
                )}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                T: {preset.temperature} • Tokens: {preset.maxTokens}
              </div>
            </motion.button>
          ))}
        </div>
      </Card>

      <Separator className="bg-slate-700" />

      {/* Temperature */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Thermometer className="h-4 w-4 text-orange-400" />
          <h3 className="text-sm font-medium text-slate-300">Temperature</h3>
          <Badge variant="outline" className="text-xs">{config.temperature}</Badge>
        </div>
        <Slider
          value={[config.temperature]}
          onValueChange={([value]) => updateConfig('temperature', value)}
          max={1.5}
          min={0.1}
          step={0.1}
          className="mb-2"
        />
        <p className="text-xs text-slate-400">
          Controls randomness. Higher values = more creative, lower = more focused.
        </p>
      </Card>

      {/* Max Tokens */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Hash className="h-4 w-4 text-green-400" />
          <h3 className="text-sm font-medium text-slate-300">Max Tokens</h3>
          <Badge variant="outline" className="text-xs">{config.maxTokens}</Badge>
        </div>
        <Slider
          value={[config.maxTokens]}
          onValueChange={([value]) => updateConfig('maxTokens', value)}
          max={500}
          min={10}
          step={10}
          className="mb-2"
        />
        <p className="text-xs text-slate-400">
          Maximum number of tokens to generate in the response.
        </p>
      </Card>

      {/* Top-K */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Target className="h-4 w-4 text-purple-400" />
          <h3 className="text-sm font-medium text-slate-300">Top-K</h3>
          <Badge variant="outline" className="text-xs">{config.topK}</Badge>
        </div>
        <Slider
          value={[config.topK]}
          onValueChange={([value]) => updateConfig('topK', value)}
          max={100}
          min={1}
          step={1}
          className="mb-2"
        />
        <p className="text-xs text-slate-400">
          Limits the number of highest probability tokens considered at each step.
        </p>
      </Card>

      {/* Top-P */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Target className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-medium text-slate-300">Top-P</h3>
          <Badge variant="outline" className="text-xs">{config.topP}</Badge>
        </div>
        <Slider
          value={[config.topP]}
          onValueChange={([value]) => updateConfig('topP', value)}
          max={1}
          min={0.1}
          step={0.05}
          className="mb-2"
        />
        <p className="text-xs text-slate-400">
          Nucleus sampling. Considers tokens with cumulative probability up to P.
        </p>
      </Card>

      {/* Current Config Summary */}
      <Card className="bg-slate-800/50 border-slate-700 p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-3">Current Configuration</h3>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-slate-400">Preset:</span>
            <span className="text-slate-300 capitalize">{config.preset}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Temperature:</span>
            <span className="text-slate-300">{config.temperature}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Max Tokens:</span>
            <span className="text-slate-300">{config.maxTokens}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Top-K:</span>
            <span className="text-slate-300">{config.topK}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Top-P:</span>
            <span className="text-slate-300">{config.topP}</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

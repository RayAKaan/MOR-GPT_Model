import { useState, useEffect } from 'react';
import { PromptInput } from '@/components/PromptInput';
import { OutputBox } from '@/components/OutputBox';
import { SettingsPanel } from '@/components/SettingsPanel';
import { HistoryPanel } from '@/components/HistoryPanel';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useToast } from '@/hooks/use-toast';
import { Settings, History, Cpu, Activity, Zap, Signal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useIsMobile } from '@/hooks/use-mobile';

interface ModelConfig {
  temperature: number;
  maxTokens: number;
  topK: number;
  topP: number;
  preset: 'creative' | 'balanced' | 'precise';
}

interface ChatMessage {
  id: string;
  prompt: string;
  response: string;
  timestamp: Date;
  config: ModelConfig;
  inferenceTime?: number;
  tokensUsed?: number;
}

const Index = () => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentResponse, setCurrentResponse] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [config, setConfig] = useState<ModelConfig>({
    temperature: 0.7,
    maxTokens: 150,
    topK: 50,
    topP: 0.9,
    preset: 'balanced'
  });

  const { toast } = useToast();
  const isMobile = useIsMobile();

  // Load history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('minigpt-history');
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Auto-close panels on mobile when generating
  useEffect(() => {
    if (isMobile && isGenerating) {
      setShowSettings(false);
      setShowHistory(false);
    }
  }, [isMobile, isGenerating]);

  // Save history to localStorage
  const saveHistory = (newHistory: ChatMessage[]) => {
    localStorage.setItem('minigpt-history', JSON.stringify(newHistory));
    setChatHistory(newHistory);
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Empty prompt",
        description: "Please enter a prompt to generate text.",
        variant: "destructive"
      });
      return;
    }

    setIsGenerating(true);
    setCurrentResponse('');

    if (isMobile) {
      setShowSettings(false);
      setShowHistory(false);
    }

    try {
      const startTime = Date.now();
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: prompt,
          temperature: config.temperature,
          max_tokens: config.maxTokens,
          top_k: config.topK,
          top_p: config.topP
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate text');
      }

      const data = await response.json();
      const inferenceTime = Date.now() - startTime;
      
      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        prompt,
        response: data.generated_text || 'Sample generated response from MiniGPT-MLOps model...',
        timestamp: new Date(),
        config,
        inferenceTime,
        tokensUsed: data.tokens_used || 42
      };

      setCurrentResponse(newMessage.response);
      const updatedHistory = [newMessage, ...chatHistory];
      saveHistory(updatedHistory);

      toast({
        title: "Generated successfully",
        description: `Response ready in ${inferenceTime}ms`,
      });

    } catch (error) {
      const mockResponse = `This is a demonstration of MiniGPT-MLOps generating text based on your prompt: "${prompt}"\n\nThe model would process this input using the transformer architecture with the configured parameters. In a production environment, this would connect to your FastAPI backend at the /predict endpoint.`;
      
      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        prompt,
        response: mockResponse,
        timestamp: new Date(),
        config,
        inferenceTime: 1200,
        tokensUsed: 85
      };

      setCurrentResponse(newMessage.response);
      const updatedHistory = [newMessage, ...chatHistory];
      saveHistory(updatedHistory);

      toast({
        title: "Demo mode",
        description: "Connect to FastAPI backend for live generation",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleHistorySelect = (message: ChatMessage) => {
    setPrompt(message.prompt);
    setCurrentResponse(message.response);
    setConfig(message.config);
    if (isMobile) {
      setShowHistory(false);
    }
  };

  const clearHistory = () => {
    localStorage.removeItem('minigpt-history');
    setChatHistory([]);
    toast({
      title: "History cleared",
      description: "All conversation history has been removed.",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 tech-grid">
      {/* Header */}
      <header className="border-b border-slate-700/30 glass-panel sticky top-0 z-50 safe-top">
        <div className="flex items-center justify-between px-4 lg:px-6 py-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Cpu className="h-6 w-6 text-slate-300" />
                <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-emerald-400/80 rounded-full status-active"></div>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-slate-200">
                  Neural Interface
                </h1>
                <p className="text-xs text-slate-500 hidden sm:block font-mono">
                  Advanced Processing System
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {!isMobile && (
              <div className="flex items-center space-x-4 text-xs text-slate-400 glass-button px-3 py-2 rounded-lg font-mono">
                <div className="flex items-center space-x-2">
                  <Signal className="h-3 w-3 text-emerald-400/80" />
                  <span>READY</span>
                </div>
                <div className="w-px h-3 bg-slate-600/50"></div>
                <div className="flex items-center space-x-2">
                  <Activity className="h-3 w-3 text-slate-400" />
                  <span>142ms</span>
                </div>
              </div>
            )}
            
            <button
              onClick={() => {
                setShowHistory(!showHistory);
                if (isMobile && !showHistory) setShowSettings(false);
              }}
              className={`glass-button p-2 rounded-lg transition-all duration-200 ${
                showHistory ? 'accent-primary text-slate-300 border-slate-500/50' : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              <History className="h-4 w-4" />
            </button>
            
            <button
              onClick={() => {
                setShowSettings(!showSettings);
                if (isMobile && !showSettings) setShowHistory(false);
              }}
              className={`glass-button p-2 rounded-lg transition-all duration-200 ${
                showSettings ? 'accent-primary text-slate-300 border-slate-500/50' : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              <Settings className="h-4 w-4" />
            </button>
            
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-73px)]">
        {/* History Panel */}
        <AnimatePresence>
          {showHistory && (
            <motion.div
              initial={{ x: isMobile ? -100 : -320, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: isMobile ? -100 : -320, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className={`${
                isMobile 
                  ? 'fixed inset-0 z-40 modal-overlay safe-left safe-right' 
                  : 'w-80 glass-panel border-r border-slate-700/30'
              }`}
            >
              <HistoryPanel
                history={chatHistory}
                onSelect={handleHistorySelect}
                onClear={clearHistory}
              />
              {isMobile && (
                <button
                  onClick={() => setShowHistory(false)}
                  className="absolute top-4 right-4 glass-button p-2 rounded-lg text-slate-400 hover:text-slate-300"
                >
                  ✕
                </button>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 p-4 lg:p-8 space-y-8 overflow-auto safe-bottom">
            <div className="max-w-4xl mx-auto space-section">
              {/* Prompt Input */}
              <motion.div
                initial={{ y: 12, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.4 }}
              >
                <PromptInput
                  value={prompt}
                  onChange={setPrompt}
                  onGenerate={handleGenerate}
                  isGenerating={isGenerating}
                />
              </motion.div>

              {/* Output */}
              {(currentResponse || isGenerating) && (
                <motion.div
                  initial={{ y: 12, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.1 }}
                >
                  <OutputBox
                    content={currentResponse}
                    isGenerating={isGenerating}
                    inferenceTime={chatHistory[0]?.inferenceTime}
                    tokensUsed={chatHistory[0]?.tokensUsed}
                  />
                </motion.div>
              )}

              {/* Welcome State */}
              {!currentResponse && !isGenerating && (
                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="text-center py-20 space-y-8"
                >
                  <div className="relative">
                    <div className="w-16 h-16 mx-auto rounded-xl glass-panel flex items-center justify-center mb-6">
                      <Cpu className="h-8 w-8 text-slate-400" />
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h2 className="text-2xl lg:text-3xl font-semibold text-slate-200">
                      System Ready
                    </h2>
                    <p className="text-slate-400 max-w-md mx-auto text-sm lg:text-base leading-relaxed">
                      Advanced language processing system initialized and ready for interaction.
                    </p>
                  </div>
                  
                  <div className="flex flex-wrap justify-center gap-4 text-xs text-slate-500 font-mono">
                    <div className="flex items-center space-x-2 glass-button px-3 py-2 rounded-lg">
                      <div className="w-1.5 h-1.5 bg-emerald-400/80 rounded-full status-active"></div>
                      <span>Online</span>
                    </div>
                    <div className="flex items-center space-x-2 glass-button px-3 py-2 rounded-lg">
                      <Activity className="h-3 w-3 text-slate-400" />
                      <span>Low Latency</span>
                    </div>
                    <div className="flex items-center space-x-2 glass-button px-3 py-2 rounded-lg">
                      <Zap className="h-3 w-3 text-slate-400" />
                      <span>GPU Ready</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ x: isMobile ? 100 : 320, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: isMobile ? 100 : 320, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className={`${
                isMobile 
                  ? 'fixed inset-0 z-40 modal-overlay safe-left safe-right' 
                  : 'w-80 glass-panel border-l border-slate-700/30'
              }`}
            >
              <SettingsPanel
                config={config}
                onConfigChange={setConfig}
              />
              {isMobile && (
                <button
                  onClick={() => setShowSettings(false)}
                  className="absolute top-4 right-4 glass-button p-2 rounded-lg text-slate-400 hover:text-slate-300"
                >
                  ✕
                </button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Index;

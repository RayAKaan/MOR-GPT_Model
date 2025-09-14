
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Copy, Download, Share, Clock, Zap, Terminal } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { motion } from 'framer-motion';

interface OutputBoxProps {
  content: string;
  isGenerating: boolean;
  inferenceTime?: number;
  tokensUsed?: number;
}

export const OutputBox = ({ content, isGenerating, inferenceTime, tokensUsed }: OutputBoxProps) => {
  const [displayedContent, setDisplayedContent] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (content && !isGenerating) {
      setIsTyping(true);
      setDisplayedContent('');
      
      let index = 0;
      const timer = setInterval(() => {
        if (index < content.length) {
          setDisplayedContent(content.slice(0, index + 1));
          index++;
        } else {
          setIsTyping(false);
          clearInterval(timer);
        }
      }, 15);

      return () => clearInterval(timer);
    }
  }, [content, isGenerating]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      toast({
        title: "Data copied to buffer",
        description: "Neural output successfully transferred.",
      });
    } catch (error) {
      toast({
        title: "Transfer failed",
        description: "Unable to access system clipboard.",
        variant: "destructive"
      });
    }
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural-output-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Export complete",
      description: "Neural data archived to local storage.",
    });
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Neural Interface Output',
          text: content,
        });
      } catch (error) {
        handleCopy();
      }
    } else {
      handleCopy();
    }
  };

  return (
    <Card className="glass-panel border-slate-700/40 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 lg:p-5 border-b border-slate-700/30">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <Terminal className="h-4 w-4 text-slate-400" />
            <h3 className="font-medium text-slate-300 font-mono">OUTPUT</h3>
          </div>
          
          {(inferenceTime || tokensUsed) && (
            <div className="flex items-center space-x-3 text-sm text-slate-400">
              {inferenceTime && (
                <div className="flex items-center space-x-2 glass-button px-2 py-1 rounded font-mono">
                  <Clock className="h-3 w-3 text-slate-400" />
                  <span>{inferenceTime}ms</span>
                </div>
              )}
              {tokensUsed && (
                <div className="flex items-center space-x-2 glass-button px-2 py-1 rounded font-mono">
                  <Zap className="h-3 w-3 text-slate-400" />
                  <span>{tokensUsed}</span>
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="glass-button text-slate-400 hover:text-slate-300 p-2"
            disabled={isGenerating}
          >
            <Copy className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            className="glass-button text-slate-400 hover:text-slate-300 p-2"
            disabled={isGenerating}
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleShare}
            className="glass-button text-slate-400 hover:text-slate-300 p-2"
            disabled={isGenerating}
          >
            <Share className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 lg:p-5">
        {isGenerating ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-4">
              <motion.div
                className="w-6 h-6 border border-slate-500/40 border-t-slate-400 rounded-full mx-auto"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <div className="space-y-2">
                <p className="text-slate-400 font-mono text-sm">Processing...</p>
                <div className="flex justify-center space-x-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1 h-1 bg-slate-400 rounded-full"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="surface-secondary rounded-lg p-4 lg:p-5 border border-slate-800/30">
            <div className="whitespace-pre-wrap text-slate-300 leading-relaxed text-sm lg:text-base">
              {displayedContent}
              {isTyping && (
                <motion.span
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                  className="text-slate-400 font-bold"
                >
                  ▊
                </motion.span>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

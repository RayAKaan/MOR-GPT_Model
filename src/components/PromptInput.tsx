
import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Loader2, Terminal } from 'lucide-react';
import { motion } from 'framer-motion';
import { useIsMobile } from '@/hooks/use-mobile';

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

export const PromptInput = ({ value, onChange, onGenerate, isGenerating }: PromptInputProps) => {
  const [charCount, setCharCount] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const isMobile = useIsMobile();

  useEffect(() => {
    setCharCount(value.length);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  useEffect(() => {
    if (textareaRef.current && !isMobile) {
      textareaRef.current.focus();
    }
  }, [isMobile]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      onGenerate();
    }
  };

  return (
    <div className="space-y-6">
      <div className="relative">
        <div className="flex items-center space-x-3 mb-4">
          <Terminal className="h-4 w-4 text-slate-400" />
          <label className="text-sm font-medium text-slate-300 font-mono">
            INPUT
          </label>
          {isFocused && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="w-1.5 h-1.5 bg-slate-400 rounded-full subtle-pulse"
            />
          )}
        </div>
        
        <div className={`relative glass-panel rounded-lg overflow-hidden transition-all duration-200 ${
          isFocused 
            ? 'accent-primary border-slate-600/50' 
            : 'border-slate-700/40 hover:border-slate-600/50'
        }`}>
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Enter your prompt..."
            className={`resize-none bg-transparent border-0 text-slate-200 placeholder:text-slate-500 focus:ring-0 focus:border-0 leading-relaxed ${
              isMobile 
                ? 'min-h-[100px] max-h-[180px] text-base p-4' 
                : 'min-h-[120px] max-h-[240px] text-sm p-5'
            }`}
            disabled={isGenerating}
          />
          
          <div className="absolute bottom-3 right-3 flex items-center space-x-3">
            <div className="text-xs text-slate-500 font-mono">
              {charCount}
            </div>
            {charCount > 400 && (
              <div className="w-1.5 h-1.5 bg-amber-400/70 rounded-full subtle-pulse"></div>
            )}
          </div>
        </div>
      </div>

      {/* Generate Button */}
      <div className="flex justify-between items-center">
        <div className="text-xs text-slate-500 font-mono">
          {!isMobile && (
            <div className="flex items-center space-x-4">
              <span>⌘+Enter to execute</span>
              <div className="w-px h-3 bg-slate-600/50"></div>
              <span className="text-slate-400">Ready</span>
            </div>
          )}
        </div>
        
        <Button
          onClick={onGenerate}
          disabled={isGenerating || !value.trim()}
          className={`btn-accent text-slate-200 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed font-mono ${
            isMobile 
              ? 'px-6 py-3 text-sm' 
              : 'px-8 py-3 text-sm'
          } ${!isGenerating && value.trim() ? 'accent-primary' : ''}`}
        >
          <div className="flex items-center">
            {isGenerating ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="mr-2"
                >
                  <Loader2 className={isMobile ? 'h-4 w-4' : 'h-4 w-4'} />
                </motion.div>
                PROCESSING
              </>
            ) : (
              <>
                <Send className={`${isMobile ? 'h-4 w-4' : 'h-4 w-4'} mr-2`} />
                EXECUTE
              </>
            )}
          </div>
        </Button>
      </div>
    </div>
  );
};

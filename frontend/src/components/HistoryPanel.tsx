
import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Search, Trash2, Clock, Copy, Star, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';
import { useToast } from '@/hooks/use-toast';

interface ChatMessage {
  id: string;
  prompt: string;
  response: string;
  timestamp: Date;
  config: {
    temperature: number;
    maxTokens: number;
    topK: number;
    topP: number;
    preset: string;
  };
  inferenceTime?: number;
  tokensUsed?: number;
}

interface HistoryPanelProps {
  history: ChatMessage[];
  onSelect: (message: ChatMessage) => void;
  onClear: () => void;
}

export const HistoryPanel = ({ history, onSelect, onClear }: HistoryPanelProps) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [favorites, setFavorites] = useState<string[]>([]);
  const { toast } = useToast();

  const filteredHistory = history.filter(message =>
    message.prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
    message.response.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleFavorite = (messageId: string) => {
    setFavorites(prev => 
      prev.includes(messageId)
        ? prev.filter(id => id !== messageId)
        : [...prev, messageId]
    );
  };

  const copyPrompt = async (prompt: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(prompt);
      toast({
        title: "Copied!",
        description: "Prompt copied to clipboard.",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy prompt.",
        variant: "destructive"
      });
    }
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(date));
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <MessageSquare className="h-5 w-5 text-blue-400" />
            <h2 className="text-lg font-semibold text-white">Chat History</h2>
          </div>
          <Badge variant="outline" className="text-xs">
            {history.length} conversations
          </Badge>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-400"
          />
        </div>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {filteredHistory.length === 0 ? (
          <div className="text-center py-8">
            <MessageSquare className="h-12 w-12 mx-auto text-slate-600 mb-3" />
            <p className="text-slate-400">
              {history.length === 0 ? 'No conversations yet' : 'No matches found'}
            </p>
          </div>
        ) : (
          filteredHistory.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Card
                className="bg-slate-800/50 border-slate-700 hover:border-slate-600 cursor-pointer transition-all duration-200 hover:shadow-lg"
                onClick={() => onSelect(message)}
              >
                <div className="p-4">
                  {/* Header with actions */}
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-3 w-3 text-slate-400" />
                      <span className="text-xs text-slate-400">
                        {formatDate(message.timestamp)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(message.id);
                        }}
                        className={`p-1 rounded-full hover:bg-slate-700 transition-colors ${
                          favorites.includes(message.id) ? 'text-yellow-400' : 'text-slate-400'
                        }`}
                      >
                        <Star className="h-3 w-3" />
                      </button>
                      <button
                        onClick={(e) => copyPrompt(message.prompt, e)}
                        className="p-1 rounded-full hover:bg-slate-700 transition-colors text-slate-400 hover:text-white"
                      >
                        <Copy className="h-3 w-3" />
                      </button>
                    </div>
                  </div>

                  {/* Prompt Preview */}
                  <div className="mb-3">
                    <p className="text-sm text-white font-medium mb-1">
                      {message.prompt.length > 100 
                        ? `${message.prompt.substring(0, 100)}...`
                        : message.prompt
                      }
                    </p>
                    <p className="text-xs text-slate-400">
                      {message.response.length > 150
                        ? `${message.response.substring(0, 150)}...`
                        : message.response
                      }
                    </p>
                  </div>

                  {/* Metadata */}
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        {message.config.preset}
                      </Badge>
                      <span className="text-slate-400">T: {message.config.temperature}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-400">
                      {message.inferenceTime && (
                        <span>{message.inferenceTime}ms</span>
                      )}
                      {message.tokensUsed && (
                        <span>{message.tokensUsed} tokens</span>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))
        )}
      </div>

      {/* Footer */}
      {history.length > 0 && (
        <div className="p-4 border-t border-slate-700">
          <Button
            variant="destructive"
            size="sm"
            onClick={onClear}
            className="w-full"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Clear All History
          </Button>
        </div>
      )}
    </div>
  );
};

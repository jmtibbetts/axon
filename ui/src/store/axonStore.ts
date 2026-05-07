import { create } from 'zustand';

export interface NeuralState {
  tick?: number;
  regions?: Record<string, number>;
  neuromod?: Record<string, number>;
  emotion?: {
    current?: string;
    valence?: number;
    arousal?: number;
    intensity?: number;
  };
  personality?: Record<string, number>;
  thoughts?: Array<{ text: string; score?: number }>;
  total_neurons?: number;
  total_connections?: number;
  new_synapses?: any[];
  conflict?: {
    dominant?: string;
    winner_set?: string[];
    score?: number;
  };
  prediction_surprise?: number;
  temporal_reward?: { mean?: number; std?: number };
  explore_eps?: number;
  cognitive_state?: {
    confidence?: number;
    uncertainty?: number;
    urgency?: number;
  };
  temporal_momentum?: number;
  temporal_depth?: number;
  critic?: { regret?: number; hesitations?: number };
  top_clusters?: Array<{ name: string; activation: number; region: string }>;
  top_routes?: any[];
  meta?: { explore_rate?: number; stability?: number; mood?: string };
  strategy_lib?: { count?: number; avg_score?: number };
  cluster_wear?: number;
  drives?: Record<string, { pressure: number; urgency: number; satisfied: boolean }>;
  beliefs?: Array<{ key: string; strength: number; valence: number; dissonance?: number }>;
  goals?: any[];
  self_model?: {
    I_am?: string[];
    I_believe?: string[];
    I_like?: string[];
    I_avoid?: string[];
    I_want?: string[];
  };
  narratives?: Record<string, number>;
  reflections?: Array<{ text: string; timestamp?: number }>;
  thought_competition?: any[];
  memory_hierarchy?: Record<string, { count: number; salience: number }>;
  interests?: Record<string, number>;
  boredom?: { level: number; factors?: string[] };
  value_scores?: Record<string, number>;
}

export interface ChatMessage {
  role: 'user' | 'axon';
  text: string;
  ts: number;
}

export interface KnowledgeIngestion {
  filename: string;
  status: 'processing' | 'done' | 'error';
  concepts?: number;
  opinions?: number;
  ts: number;
  error?: string;
}

interface AxonStore {
  connected: boolean;
  engineRunning: boolean;
  neuralState: NeuralState;
  lastTick: number;
  messages: ChatMessage[];
  thinking: boolean;
  visionFrame: string | null;
  faceData: any;
  hebbianEvents: any[];
  memoryEvents: any[];
  regionSpikes: any[];
  logs: any[];
  lmStatus: any;
  activeTab: string;
  ingestions: KnowledgeIngestion[];
  thoughtCompetition: any[];
  surpriseEvents: any[];
  reflections: any[];
  autonomousMode: boolean;
  voiceSpeaking: boolean;
  audioEmotion: { emotion?: string; energy?: number; valence?: number } | null;
  micVolume: number;
  userProfile: any;
  // historical series for charts
  rewardHistory: number[];
  surpriseHistory: number[];
  regionHistory: Record<string, number[]>;
  nmHistory: Record<string, number[]>;
  set: (partial: Partial<AxonStore> | ((state: AxonStore) => Partial<AxonStore>)) => void;
}

export const useAxonStore = create<AxonStore>((set) => ({
  connected: false,
  engineRunning: false,
  neuralState: {},
  lastTick: 0,
  messages: [],
  thinking: false,
  visionFrame: null,
  faceData: null,
  hebbianEvents: [],
  memoryEvents: [],
  regionSpikes: [],
  logs: [],
  lmStatus: null,
  activeTab: 'overview',
  ingestions: [],
  thoughtCompetition: [],
  surpriseEvents: [],
  reflections: [],
  autonomousMode: false,
  voiceSpeaking: false,
  audioEmotion: null,
  micVolume: 0,
  userProfile: null,
  rewardHistory: [],
  surpriseHistory: [],
  regionHistory: {},
  nmHistory: {},
  set: (partial) =>
    set((state) =>
      typeof partial === 'function' ? partial(state) : { ...state, ...partial }
    ),
}));

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
  meta?: { explore_rate?: number; stability?: number };
  strategy_lib?: { count?: number; avg_score?: number };
  cluster_wear?: number;
}

export interface ChatMessage {
  role: 'user' | 'axon';
  text: string;
  ts: number;
}

interface AxonStore {
  // connection
  connected: boolean;
  // neural
  neuralState: NeuralState;
  lastTick: number;
  // chat
  messages: ChatMessage[];
  thinking: boolean;
  // vision
  visionFrame: string | null;
  faceData: any;
  // events
  hebbianEvents: any[];
  memoryEvents: any[];
  regionSpikes: any[];
  logs: any[];
  // lm
  lmStatus: any;
  // ui
  activeTab: string;
  // setter
  set: (partial: Partial<AxonStore> | ((state: AxonStore) => Partial<AxonStore>)) => void;
}

export const useAxonStore = create<AxonStore>((set) => ({
  connected: false,
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
  activeTab: 'brain',
  set: (partial) =>
    set((state) =>
      typeof partial === 'function' ? partial(state) : { ...state, ...partial }
    ),
}));

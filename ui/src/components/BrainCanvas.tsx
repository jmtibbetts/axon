import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useAxonStore } from '../store/axonStore';
import type { NeuralState } from '../store/axonStore';

// ── Region colour map ──────────────────────────────────────────────────────
const REGION_COLORS: Record<string, string> = {
  prefrontal:    '#6366f1',
  hippocampus:   '#22d3ee',
  amygdala:      '#f43f5e',
  visual:        '#a3e635',
  auditory:      '#fb923c',
  language:      '#e879f9',
  thalamus:      '#fbbf24',
  cerebellum:    '#34d399',
  association:   '#818cf8',
  default_mode:  '#94a3b8',
  social:        '#f472b6',
  metacognition: '#c084fc',
};

const REGION_POSITIONS: Record<string, [number, number, number]> = {
  prefrontal:    [0, 0.6, 0.7],
  hippocampus:   [0.4, -0.2, 0],
  amygdala:      [0.5, -0.3, 0.2],
  visual:        [0, -0.5, -0.8],
  auditory:      [0.7, 0.1, 0],
  language:      [-0.6, 0.1, 0.3],
  thalamus:      [0, 0, 0],
  cerebellum:    [0, -0.7, -0.5],
  association:   [-0.3, 0.4, -0.3],
  default_mode:  [0, 0.2, -0.6],
  social:        [0.3, 0.5, 0.2],
  metacognition: [-0.2, 0.7, 0.1],
};

const SPHERE_RADIUS = 2.2;
const NUM_PARTICLES = 4000;

// Emotion → ambient light color
const EMOTION_COLORS: Record<string, string> = {
  happy:     '#4ade80',
  excited:   '#fbbf24',
  curious:   '#22d3ee',
  calm:      '#6366f1',
  neutral:   '#475569',
  sad:       '#3b82f6',
  anxious:   '#f97316',
  angry:     '#ef4444',
  bored:     '#374151',
  focused:   '#a855f7',
  surprised: '#facc15',
  afraid:    '#dc2626',
};

// ── Shared refs passed down from Scene ──────────────────────────────────────
interface SceneRefs {
  spikeFlashes: React.MutableRefObject<Record<string, number>>; // region → flash intensity (0..1, decays)
  hebbianFlash: React.MutableRefObject<{ src: string; dst: string; t: number } | null>;
  surpriseFlash: React.MutableRefObject<number>; // 0..1, decays
  rewardFlash:   React.MutableRefObject<number>; // +ve green, -ve red, decays
  thinkingPulse: React.MutableRefObject<number>; // 0..1 when thinking
}

// ── Particle system ────────────────────────────────────────────────────────
function NeuronParticles({ ns, refs }: { ns: NeuralState; refs: SceneRefs }) {
  const meshRef = useRef<THREE.Points>(null!);

  const { positions, baseColors, regionIdx } = useMemo(() => {
    const positions  = new Float32Array(NUM_PARTICLES * 3);
    const baseColors = new Float32Array(NUM_PARTICLES * 3);
    const regionIdx  = new Int32Array(NUM_PARTICLES);
    const regionKeys = Object.keys(REGION_POSITIONS);

    for (let i = 0; i < NUM_PARTICLES; i++) {
      const rk = regionKeys[i % regionKeys.length];
      const [rx, ry, rz] = REGION_POSITIONS[rk];
      const theta = Math.acos(2 * Math.random() - 1);
      const phi   = Math.random() * Math.PI * 2;
      const sx = Math.sin(theta) * Math.cos(phi);
      const sy = Math.sin(theta) * Math.sin(phi);
      const sz = Math.cos(theta);
      const t = 0.35 + Math.random() * 0.45;
      positions[i * 3]     = (sx * (1 - t) + rx * t) * SPHERE_RADIUS;
      positions[i * 3 + 1] = (sy * (1 - t) + ry * t) * SPHERE_RADIUS;
      positions[i * 3 + 2] = (sz * (1 - t) + rz * t) * SPHERE_RADIUS;
      const col = new THREE.Color(REGION_COLORS[rk] ?? '#ffffff');
      baseColors[i * 3]     = col.r;
      baseColors[i * 3 + 1] = col.g;
      baseColors[i * 3 + 2] = col.b;
      regionIdx[i] = regionKeys.indexOf(rk);
    }
    return { positions, baseColors, regionIdx };
  }, []);

  const activationRef     = useRef<Record<string, number>>({});
  const neuromodRef       = useRef<Record<string, number>>({});
  const thinkingRef       = useRef(false);
  const exploreEpsRef     = useRef(0.1);
  const surpriseLevelRef  = useRef(0);
  const conflictScoreRef  = useRef(0);

  useEffect(() => { if (ns.regions)         activationRef.current = ns.regions; },         [ns.regions]);
  useEffect(() => { if (ns.neuromod)        neuromodRef.current   = ns.neuromod as Record<string,number>; }, [ns.neuromod]);
  useEffect(() => { if (ns.explore_eps != null) exploreEpsRef.current = ns.explore_eps; },  [ns.explore_eps]);
  useEffect(() => { if (ns.prediction_surprise != null) surpriseLevelRef.current = ns.prediction_surprise; }, [ns.prediction_surprise]);
  useEffect(() => { if (ns.conflict?.score != null)     conflictScoreRef.current = ns.conflict!.score!; },    [ns.conflict]);

  const thinking = useAxonStore((s) => s.thinking);
  useEffect(() => { thinkingRef.current = thinking; }, [thinking]);

  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const geo  = meshRef.current.geometry;
    const colAttr = geo.attributes.color as THREE.BufferAttribute;
    const posAttr = geo.attributes.position as THREE.BufferAttribute;
    const regionKeys = Object.keys(REGION_POSITIONS);

    const dopamine    = neuromodRef.current['dopamine']    ?? 0.3;
    const serotonin   = neuromodRef.current['serotonin']   ?? 0.3;
    const norepineph  = neuromodRef.current['norepinephrine'] ?? 0.2;
    const gaba        = neuromodRef.current['gaba']        ?? 0.4;
    const eps         = exploreEpsRef.current;
    const surprise    = Math.min(1, surpriseLevelRef.current * 3);
    const conflict    = Math.min(1, conflictScoreRef.current);
    const isThinking  = thinkingRef.current;

    // global modifiers from neuromod
    const energyMult  = 0.6 + dopamine * 0.8 + norepineph * 0.4;   // how bright
    const speedMult   = 0.8 + dopamine * 1.2 + norepineph * 0.6;    // pulse speed
    const calmMult    = serotonin * 0.4 + gaba * 0.3;               // dampen noise
    const scatterMult = Math.max(0, eps - 0.1) * 2.0;               // how scattered

    for (let i = 0; i < NUM_PARTICLES; i++) {
      const rk   = regionKeys[regionIdx[i] % regionKeys.length];
      const act  = activationRef.current[rk] ?? 0.1;
      const spk  = refs.spikeFlashes.current[rk] ?? 0;

      // Base pulse driven by activation + neuromod energy
      const freq = (1.0 + act * 2.5 + dopamine * 1.5) * speedMult;
      const basePulse = (0.3 + act * 0.5 + spk * 0.5) * energyMult
        + Math.sin(t.current * freq + i * 0.02) * 0.18 * (act + 0.1) * (1 - calmMult * 0.5);

      // Surprise flash: white-hot burst across all particles
      const surpriseAdd = surprise > 0.3
        ? Math.max(0, surprise - 0.3) * Math.exp(-t.current % 0.5 * 8) * 1.5
        : 0;

      // Conflict: adds red-shift and flicker to high-activation regions
      const conflictFlicker = conflict > 0.4
        ? Math.abs(Math.sin(t.current * 18 + i)) * conflict * act * 0.4
        : 0;

      // Exploration scatter: particles jitter more when eps is high
      if (scatterMult > 0.05) {
        const origX = positions[i * 3];
        const origY = positions[i * 3 + 1];
        const origZ = positions[i * 3 + 2];
        const jitter = scatterMult * 0.04;
        posAttr.setXYZ(
          i,
          origX + (Math.random() - 0.5) * jitter,
          origY + (Math.random() - 0.5) * jitter,
          origZ + (Math.random() - 0.5) * jitter,
        );
        posAttr.needsUpdate = true;
      }

      // Thinking: all particles pulse in a slow ring wave
      const thinkAdd = isThinking
        ? Math.abs(Math.sin(t.current * 3 + i * 0.005)) * 0.3
        : 0;

      const brightness = Math.max(0, Math.min(2.5, basePulse + surpriseAdd + thinkAdd));

      const br = baseColors[i * 3];
      const bg = baseColors[i * 3 + 1];
      const bb = baseColors[i * 3 + 2];

      // conflict adds red, spike adds white
      const r = br * brightness + conflictFlicker * 0.6 + spk * 0.3;
      const g = bg * brightness                           + spk * 0.3;
      const b = bb * brightness                           + spk * 0.3;

      colAttr.setXYZ(i, Math.min(1, r), Math.min(1, g), Math.min(1, b));
    }

    colAttr.needsUpdate = true;

    // Rotation: dopamine = faster, serotonin = slower/calmer
    const rotSpeed = (0.02 + dopamine * 0.05 - serotonin * 0.01 + (isThinking ? 0.04 : 0));
    meshRef.current.rotation.y += delta * Math.max(0.005, rotSpeed);
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry
        onUpdate={(self) => {
          self.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
          self.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(baseColors), 3));
        }}
      />
      <pointsMaterial size={0.042} vertexColors sizeAttenuation transparent opacity={0.88} />
    </points>
  );
}

// ── Synapse arcs ──────────────────────────────────────────────────────────
function SynapseArcs({ ns, refs }: { ns: NeuralState; refs: SceneRefs }) {
  const groupRef = useRef<THREE.Group>(null!);

  const lines = useMemo(() => {
    const routes = ns.top_routes ?? [];
    return routes.slice(0, 10).map((r: any, i: number) => {
      const srcPos = REGION_POSITIONS[r.src_region ?? r.src] ?? [0, 0, 0];
      const dstPos = REGION_POSITIONS[r.dst_region ?? r.dst] ?? [0, 0, 0];
      const mid: [number, number, number] = [
        (srcPos[0] + dstPos[0]) / 2,
        (srcPos[1] + dstPos[1]) / 2 + 0.7,
        (srcPos[2] + dstPos[2]) / 2,
      ];
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(...srcPos).multiplyScalar(SPHERE_RADIUS * 0.82),
        new THREE.Vector3(...mid).multiplyScalar(SPHERE_RADIUS * 0.82),
        new THREE.Vector3(...dstPos).multiplyScalar(SPHERE_RADIUS * 0.82),
      );
      const pts = curve.getPoints(50);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const w   = Math.min(1, (r.weight ?? 0.3) * 2.5);
      const col = new THREE.Color(REGION_COLORS[r.src_region ?? r.src] ?? '#6366f1');
      const mat = new THREE.LineBasicMaterial({
        color: col, transparent: true, opacity: w * 0.55,
      });
      return { geo, mat, key: i, srcRegion: r.src_region ?? r.src, dstRegion: r.dst_region ?? r.dst, weight: w };
    });
  }, [ns.top_routes]);

  // Hebbian flash: light up specific arc
  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    if (!groupRef.current) return;
    const hf = refs.hebbianFlash.current;
    groupRef.current.children.forEach((child, i) => {
      const line = child as THREE.Line;
      if (!line.material) return;
      const mat = line.material as THREE.LineBasicMaterial;
      const l   = lines[i];
      if (!l) return;
      const isHebbian = hf && (hf.src === l.srcRegion || hf.src === l.dstRegion);
      const flashAmt  = isHebbian ? Math.max(0, 1 - (t.current - hf!.t) * 2) : 0;
      mat.opacity = Math.min(0.9, l.weight * 0.55 + flashAmt * 0.8);
      if (flashAmt > 0.1) {
        mat.color.setHex(0xffffff);
      } else {
        mat.color.set(REGION_COLORS[l.srcRegion] ?? '#6366f1');
      }
    });
  });

  return (
    <group ref={groupRef}>
      {lines.map(({ geo, mat, key }) => (
        <primitive key={key} object={new THREE.Line(geo, mat)} />
      ))}
    </group>
  );
}

// ── Region orbs ────────────────────────────────────────────────────────────
function RegionOrbs({ ns, refs }: { ns: NeuralState; refs: SceneRefs }) {
  const groupRef = useRef<THREE.Group>(null!);
  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    if (!groupRef.current) return;
    const regionKeys = Object.keys(REGION_POSITIONS);
    groupRef.current.children.forEach((child, i) => {
      const mesh = child as THREE.Mesh;
      const rk   = regionKeys[i];
      if (!rk) return;
      const act  = (ns.regions ?? {})[rk] ?? 0.05;
      const spk  = refs.spikeFlashes.current[rk] ?? 0;
      const cog  = ns.cognitive_state ?? {};
      const urgency = cog.urgency ?? 0;

      // Scale: base size + activation + spike burst + urgency
      const scale = 0.055 + act * 0.14 + spk * 0.12 + urgency * 0.02
        + Math.sin(t.current * (1.5 + act * 3) + i * 0.5) * 0.008 * (act + 0.1);
      mesh.scale.setScalar(Math.max(0.02, scale));

      const mat = mesh.material as THREE.MeshStandardMaterial;
      // Emissive: activation + spike + conflict shimmer
      const conflictScore = ns.conflict?.score ?? 0;
      const shimmer = conflictScore > 0.5
        ? Math.abs(Math.sin(t.current * 12 + i * 0.7)) * conflictScore * 0.5
        : 0;
      mat.emissiveIntensity = Math.min(3, 0.25 + act * 1.6 + spk * 1.5 + shimmer);

      // Spike: briefly whiten the orb
      if (spk > 0.1) {
        mat.emissive.setHex(0xffffff);
        mat.emissive.lerp(new THREE.Color(REGION_COLORS[rk] ?? '#ffffff'), 1 - spk);
      } else {
        mat.emissive.set(REGION_COLORS[rk] ?? '#ffffff');
      }
    });
  });

  return (
    <group ref={groupRef}>
      {Object.entries(REGION_POSITIONS).map(([rk, pos]) => {
        const col = new THREE.Color(REGION_COLORS[rk] ?? '#ffffff');
        return (
          <mesh
            key={rk}
            position={pos.map((v) => v * SPHERE_RADIUS * 0.85) as [number, number, number]}
          >
            <sphereGeometry args={[1, 14, 14]} />
            <meshStandardMaterial
              color={col}
              emissive={col}
              emissiveIntensity={0.5}
              transparent
              opacity={0.75}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ── Ambient + point lights driven by emotion / neuromod ──────────────────
function DynamicLights({ ns }: { ns: NeuralState }) {
  const ambientRef    = useRef<THREE.AmbientLight>(null!);
  const pointRef1     = useRef<THREE.PointLight>(null!);
  const pointRef2     = useRef<THREE.PointLight>(null!);
  const rewardFlashRef = useRef(0);
  const lastReward     = useRef(0);

  const reward   = (ns.temporal_reward as any)?.mean ?? 0;
  const surprise = ns.prediction_surprise ?? 0;
  useEffect(() => {
    if (Math.abs(reward - lastReward.current) > 0.15) {
      rewardFlashRef.current = reward > lastReward.current ? 1 : -1;
      lastReward.current = reward;
    }
  }, [reward]);

  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    rewardFlashRef.current *= Math.max(0, 1 - delta * 2);

    const emotion  = ns.emotion?.current?.toLowerCase() ?? 'neutral';
    const emoCol   = EMOTION_COLORS[emotion] ?? '#475569';
    const nm       = ns.neuromod as Record<string,number> ?? {};
    const dopamine = nm['dopamine']  ?? 0.3;
    const serotonin = nm['serotonin'] ?? 0.3;

    if (ambientRef.current) {
      // Ambient softly tracks emotion color
      ambientRef.current.color.lerp(new THREE.Color(emoCol), 0.03);
      ambientRef.current.intensity = 0.25 + serotonin * 0.15;
    }
    if (pointRef1.current) {
      // Primary light: dopamine brightness + surprise flash
      const surpFlash = surprise > 0.4 ? Math.exp(-((t.current % 1) * 6)) * 2 : 0;
      const rewFlash  = rewardFlashRef.current > 0 ? rewardFlashRef.current * 1.5 : 0;
      pointRef1.current.intensity = 1.2 + dopamine * 1.5 + surpFlash + rewFlash;
      pointRef1.current.color.lerp(new THREE.Color(emoCol), 0.02);
    }
    if (pointRef2.current) {
      // Secondary light: negative reward → red tint
      const negFlash = rewardFlashRef.current < 0 ? Math.abs(rewardFlashRef.current) * 1.2 : 0;
      pointRef2.current.intensity = 0.6 + negFlash;
      if (negFlash > 0.1) {
        pointRef2.current.color.lerp(new THREE.Color('#ef4444'), 0.1);
      } else {
        pointRef2.current.color.lerp(new THREE.Color('#22d3ee'), 0.02);
      }
    }
  });

  return (
    <>
      <ambientLight ref={ambientRef} intensity={0.3} color="#4444aa" />
      <pointLight ref={pointRef1} position={[5, 5, 5]}    intensity={1.5} color="#6366f1" />
      <pointLight ref={pointRef2} position={[-5, -3, -5]} intensity={0.8} color="#22d3ee" />
    </>
  );
}

// ── Outer glass sphere ────────────────────────────────────────────────────
function GlassSphere({ ns }: { ns: NeuralState }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    const urgency  = ns.cognitive_state?.urgency ?? 0;
    const conflict = ns.conflict?.score ?? 0;
    // Glass gets slightly less transparent when urgency/conflict is high
    mat.opacity = 0.03 + urgency * 0.04 + conflict * 0.03;
  });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[SPHERE_RADIUS + 0.05, 48, 48]} />
      <meshStandardMaterial color="#6366f1" transparent opacity={0.04} wireframe={false} side={THREE.BackSide} />
    </mesh>
  );
}

function WireFrame({ ns }: { ns: NeuralState }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const mat = meshRef.current.material as THREE.MeshBasicMaterial;
    // Wireframe pulses opacity with confidence
    const conf = ns.cognitive_state?.confidence ?? 0.5;
    mat.opacity = 0.12 + conf * 0.18 + Math.sin(t.current * 0.5) * 0.04;
  });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[SPHERE_RADIUS + 0.06, 16, 12]} />
      <meshBasicMaterial color="#1e1b4b" wireframe transparent opacity={0.2} />
    </mesh>
  );
}

// ── Thinking ring ─────────────────────────────────────────────────────────
function ThinkingRing() {
  const groupRef  = useRef<THREE.Group>(null!);
  const thinking  = useAxonStore((s) => s.thinking);
  const thinkRef  = useRef(false);
  const alphaRef  = useRef(0);
  useEffect(() => { thinkRef.current = thinking; }, [thinking]);

  const ringGeo = useMemo(() => new THREE.TorusGeometry(SPHERE_RADIUS + 0.3, 0.015, 8, 64), []);
  const ringMat = useMemo(() => new THREE.MeshBasicMaterial({ color: '#6366f1', transparent: true, opacity: 0 }), []);

  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    if (!groupRef.current) return;
    // Fade in/out
    const target = thinkRef.current ? 0.7 : 0;
    alphaRef.current += (target - alphaRef.current) * delta * 3;
    ringMat.opacity = alphaRef.current;
    ringMat.color.set(thinkRef.current ? '#a5b4fc' : '#6366f1');
    // Orbit the ring
    groupRef.current.rotation.x += delta * 0.8;
    groupRef.current.rotation.z += delta * 0.5;
  });

  return (
    <group ref={groupRef}>
      <mesh geometry={ringGeo} material={ringMat} />
    </group>
  );
}

// ── Scene ──────────────────────────────────────────────────────────────────
function Scene() {
  const ns             = useAxonStore((s) => s.neuralState);
  const regionSpikes   = useAxonStore((s) => s.regionSpikes);
  const hebbianEvents  = useAxonStore((s) => s.hebbianEvents);

  // Shared flash state refs (no re-render — animation loop reads directly)
  const spikeFlashes  = useRef<Record<string, number>>({});
  const hebbianFlash  = useRef<{ src: string; dst: string; t: number } | null>(null);
  const surpriseFlash = useRef(0);
  const rewardFlash   = useRef(0);
  const thinkingPulse = useRef(0);
  const { clock }     = useThree();

  // React to new region spikes
  const lastSpikeLen = useRef(0);
  useEffect(() => {
    if (regionSpikes.length > lastSpikeLen.current) {
      const latest = regionSpikes[0];
      if (latest?.region) {
        spikeFlashes.current[latest.region] = Math.min(1, (latest.activation ?? 0.6) * 1.5);
      }
      lastSpikeLen.current = regionSpikes.length;
    }
  }, [regionSpikes]);

  // React to new hebbian events
  const lastHebLen = useRef(0);
  useEffect(() => {
    if (hebbianEvents.length > lastHebLen.current) {
      const ev = hebbianEvents[0];
      if (ev) {
        hebbianFlash.current = {
          src: ev.src ?? ev.a ?? '',
          dst: ev.dst ?? ev.b ?? '',
          t:   clock.getElapsedTime(),
        };
      }
      lastHebLen.current = hebbianEvents.length;
    }
  }, [hebbianEvents, clock]);

  // Decay spike flashes every frame
  useFrame((_, delta) => {
    Object.keys(spikeFlashes.current).forEach((rk) => {
      spikeFlashes.current[rk] = Math.max(0, spikeFlashes.current[rk] - delta * 2.5);
    });
  });

  const refs: SceneRefs = { spikeFlashes, hebbianFlash, surpriseFlash, rewardFlash, thinkingPulse };

  return (
    <>
      <DynamicLights ns={ns} />
      <NeuronParticles ns={ns} refs={refs} />
      <SynapseArcs    ns={ns} refs={refs} />
      <RegionOrbs     ns={ns} refs={refs} />
      <GlassSphere    ns={ns} />
      <WireFrame      ns={ns} />
      <ThinkingRing />
      <OrbitControls enablePan={false} minDistance={3} maxDistance={9} autoRotate autoRotateSpeed={0.3} />
    </>
  );
}

// ── Exported component ─────────────────────────────────────────────────────
export default function BrainCanvas() {
  return (
    <div style={{ width: '100%', height: '100%', background: '#020205' }}>
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 55 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#020205']} />
        <Scene />
      </Canvas>
    </div>
  );
}

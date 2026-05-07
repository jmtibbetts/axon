import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useAxonStore } from '../store/axonStore';
import type { NeuralState } from '../store/axonStore';

// ── All 14 real AXON regions (restored from fabric) ─────────────────────
const REGION_DEFS: {
  key: string;
  label: string;
  color: string;
  pos: [number, number, number];  // normalised unit sphere
  particles: number;
  spread: number;
}[] = [
  // ── Prefrontal complex (top-front) ──
  { key: 'planning_cortex',       label: 'Planning',     color: '#6366f1', pos: [-0.18, 0.72, 0.60], particles: 280, spread: 0.28 },
  { key: 'working_memory_cortex', label: 'Working Mem',  color: '#818cf8', pos: [ 0.18, 0.72, 0.60], particles: 260, spread: 0.26 },
  { key: 'inhibitory_cortex',     label: 'Inhibitory',   color: '#a5b4fc', pos: [ 0.00, 0.68, 0.68], particles: 200, spread: 0.22 },
  // ── Limbic (centre, buried) ──
  { key: 'hippocampus',           label: 'Hippocampus',  color: '#22d3ee', pos: [ 0.42, -0.15, 0.08], particles: 260, spread: 0.22 },
  { key: 'amygdala',              label: 'Amygdala',     color: '#f43f5e', pos: [ 0.46, -0.28, 0.22], particles: 220, spread: 0.20 },
  { key: 'thalamus',              label: 'Thalamus',     color: '#fbbf24', pos: [ 0.00,  0.00, 0.00], particles: 240, spread: 0.18 },
  // ── Sensory cortex ──
  { key: 'visual',                label: 'Visual',       color: '#a3e635', pos: [ 0.05, -0.55, -0.82], particles: 240, spread: 0.26 },
  { key: 'auditory',              label: 'Auditory',     color: '#fb923c', pos: [ 0.72,  0.08, -0.10], particles: 220, spread: 0.22 },
  // ── Motor/language ──
  { key: 'language',              label: 'Language',     color: '#e879f9', pos: [-0.62,  0.12, 0.32], particles: 240, spread: 0.24 },
  { key: 'cerebellum',            label: 'Cerebellum',   color: '#34d399', pos: [ 0.00, -0.72, -0.52], particles: 200, spread: 0.26 },
  // ── Higher-order ──
  { key: 'default_mode',          label: 'Default Mode', color: '#94a3b8', pos: [-0.02,  0.22, -0.78], particles: 200, spread: 0.24 },
  { key: 'association',           label: 'Association',  color: '#f472b6', pos: [-0.38,  0.38, -0.30], particles: 200, spread: 0.24 },
  { key: 'social',                label: 'Social',       color: '#38bdf8', pos: [ 0.32,  0.52, 0.18], particles: 180, spread: 0.20 },
  { key: 'metacognition',         label: 'Metacog',      color: '#c084fc', pos: [-0.22,  0.78, 0.05], particles: 180, spread: 0.20 },
];

// backward compat: "prefrontal" → average of the 3 PFC keys
function getRegionAct(regions: Record<string, number>, key: string): number {
  if (key in regions) return regions[key];
  if (key === 'planning_cortex' || key === 'working_memory_cortex' || key === 'inhibitory_cortex') {
    // Try prefrontal fallback
    return regions['prefrontal'] ?? 0.05;
  }
  return 0.05;
}

const SPHERE_R = 2.1;
const TOTAL_PARTICLES = REGION_DEFS.reduce((s, r) => s + r.particles, 0);

// ── Emotion → ambient color ───────────────────────────────────────────────
const EMO_COLORS: Record<string, string> = {
  happy: '#4ade80', excited: '#fbbf24', curious: '#22d3ee', calm: '#6366f1',
  neutral: '#475569', sad: '#3b82f6', anxious: '#f97316', angry: '#ef4444',
  bored: '#374151', focused: '#a855f7', surprised: '#facc15', afraid: '#dc2626',
};

// ─────────────────────────────────────────────────────────────────────────
// 1. PARTICLE CLOUD — per-region clusters
// ─────────────────────────────────────────────────────────────────────────
interface SpikeRefs {
  spikes: React.MutableRefObject<Record<string, number>>;
}

function NeuronParticles({ ns, refs }: { ns: NeuralState; refs: SpikeRefs }) {
  const meshRef = useRef<THREE.Points>(null!);

  // Build initial geometry — each region gets its own cluster of particles
  const { positions, baseColors, regionMap } = useMemo(() => {
    const total = TOTAL_PARTICLES;
    const positions  = new Float32Array(total * 3);
    const baseColors = new Float32Array(total * 3);
    const regionMap  = new Uint8Array(total); // index into REGION_DEFS

    let idx = 0;
    REGION_DEFS.forEach((rd, ri) => {
      const [cx, cy, cz] = rd.pos;
      const col = new THREE.Color(rd.color);
      for (let p = 0; p < rd.particles; p++) {
        // Gaussian-ish cluster around the region centre
        const u1 = Math.random(), u2 = Math.random(), u3 = Math.random();
        const theta = Math.acos(2 * u1 - 1);
        const phi   = u2 * Math.PI * 2;
        const r     = Math.pow(u3, 0.33) * rd.spread;
        const nx    = Math.sin(theta) * Math.cos(phi) * r;
        const ny    = Math.sin(theta) * Math.sin(phi) * r;
        const nz    = Math.cos(theta) * r;

        // Place on sphere surface, offset by cluster centre direction
        const wx = cx + nx;
        const wy = cy + ny;
        const wz = cz + nz;
        // Normalise back onto sphere surface with some depth variation
        const len = Math.sqrt(wx*wx + wy*wy + wz*wz);
        const d = SPHERE_R * (0.7 + 0.3 * (Math.random()));
        positions[idx * 3]     = (wx / len) * d;
        positions[idx * 3 + 1] = (wy / len) * d;
        positions[idx * 3 + 2] = (wz / len) * d;

        baseColors[idx * 3]     = col.r;
        baseColors[idx * 3 + 1] = col.g;
        baseColors[idx * 3 + 2] = col.b;
        regionMap[idx] = ri;
        idx++;
      }
    });
    return { positions, baseColors, regionMap };
  }, []);

  // Live references (avoid re-renders)
  const actRef   = useRef<Record<string, number>>({});
  const nmRef    = useRef<Record<string, number>>({});
  const thinkRef = useRef(false);
  const thinking = useAxonStore((s) => s.thinking);
  useEffect(() => { if (ns.regions)  actRef.current  = ns.regions as Record<string,number>; }, [ns.regions]);
  useEffect(() => { if (ns.neuromod) nmRef.current   = ns.neuromod as Record<string,number>; }, [ns.neuromod]);
  useEffect(() => { thinkRef.current = thinking; }, [thinking]);

  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const geo      = meshRef.current.geometry;
    const colAttr  = geo.attributes.color as THREE.BufferAttribute;

    const da   = nmRef.current['dopamine']        ?? 0.3;
    const ne   = nmRef.current['norepinephrine']  ?? 0.2;
    const ser  = nmRef.current['serotonin']       ?? 0.3;
    const gaba = nmRef.current['gaba']            ?? 0.4;
    const isThinking = thinkRef.current;
    const conflict = (ns.conflict as any)?.score ?? (ns.conflict as any)?.dominance_mean ?? 0;

    const energyBase = 0.5 + da * 0.7 + ne * 0.4;
    const calm       = ser * 0.3 + gaba * 0.2;

    for (let i = 0; i < TOTAL_PARTICLES; i++) {
      const ri   = regionMap[i];
      const rd   = REGION_DEFS[ri];
      const act  = getRegionAct(actRef.current, rd.key);
      const spk  = refs.spikes.current[rd.key] ?? 0;

      // Per-particle frequency varies slightly so they don't all pulse in unison
      const phase = i * 0.031;
      const freq  = 1.2 + act * 3.0 + da * 1.5;
      const pulse = (act + 0.08 + spk * 0.6) * energyBase
        + Math.sin(t.current * freq + phase) * 0.15 * (act + 0.05) * (1 - calm * 0.4);

      // Thinking: slow tidal wave
      const thinkAdd = isThinking
        ? Math.abs(Math.sin(t.current * 2.5 + i * 0.004)) * 0.35
        : 0;

      // Conflict: red flicker on high-activation regions
      const confFlicker = conflict > 0.35 && act > 0.3
        ? Math.abs(Math.sin(t.current * 16 + i)) * conflict * act * 0.5
        : 0;

      const brightness = Math.max(0.02, Math.min(2.5, pulse + thinkAdd));

      const br = baseColors[i * 3];
      const bg = baseColors[i * 3 + 1];
      const bb = baseColors[i * 3 + 2];

      colAttr.setXYZ(i,
        Math.min(1, br * brightness + confFlicker * 0.5 + spk * 0.25),
        Math.min(1, bg * brightness                       + spk * 0.25),
        Math.min(1, bb * brightness                       + spk * 0.25),
      );
    }
    colAttr.needsUpdate = true;

    // Slow rotation driven by dopamine
    meshRef.current.rotation.y += delta * (0.018 + da * 0.04 + (isThinking ? 0.03 : 0));
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry
        onUpdate={(self) => {
          self.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
          self.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(baseColors), 3));
        }}
      />
      <pointsMaterial size={0.038} vertexColors sizeAttenuation transparent opacity={0.92} />
    </points>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 2. REGION ORBS — individual animated sphere + label per region
// ─────────────────────────────────────────────────────────────────────────
function RegionOrb({ rd, ns, spikesRef }: {
  rd: typeof REGION_DEFS[0];
  ns: NeuralState;
  spikesRef: React.MutableRefObject<Record<string, number>>;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const t = useRef(0);
  const [actPct, setActPct] = React.useState(5);

  const pos: [number, number, number] = [
    rd.pos[0] * SPHERE_R * 0.88,
    rd.pos[1] * SPHERE_R * 0.88,
    rd.pos[2] * SPHERE_R * 0.88,
  ];
  const col = useMemo(() => new THREE.Color(rd.color), [rd.color]);

  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const act = getRegionAct((ns.regions ?? {}) as Record<string,number>, rd.key);
    const spk = spikesRef.current[rd.key] ?? 0;
    const conflict = (ns.conflict as any)?.score ?? 0;

    const scale = 0.048 + act * 0.18 + spk * 0.16
      + Math.sin(t.current * (1.4 + act * 3) + rd.pos[0] * 5) * 0.007 * (act + 0.05);
    meshRef.current.scale.setScalar(Math.max(0.02, scale));

    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    const shimmer = conflict > 0.5 && act > 0.3
      ? Math.abs(Math.sin(t.current * 14 + rd.pos[1] * 8)) * conflict * 0.45
      : 0;
    mat.emissiveIntensity = Math.min(4.0, 0.15 + act * 2.5 + spk * 2.0 + shimmer);

    if (spk > 0.05) {
      mat.emissive.setHex(0xffffff);
      mat.emissive.lerp(col, 1 - spk * 0.8);
    } else {
      mat.emissive.copy(col);
    }

    // Update label pct at low frequency to avoid React re-render storm
    if (Math.floor(t.current * 3) !== Math.floor((t.current - delta) * 3)) {
      setActPct(Math.round(act * 100));
    }
  });

  return (
    <mesh ref={meshRef} position={pos}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshStandardMaterial
        color={col}
        emissive={col}
        emissiveIntensity={0.35}
        transparent
        opacity={0.72}
      />
      <Html
        center
        position={[0, 1.5, 0]}
        style={{ pointerEvents: 'none', userSelect: 'none' }}
        distanceFactor={6.5}
        zIndexRange={[10, 10]}
      >
        <div style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
          <div style={{
            fontSize: 9, fontFamily: 'monospace', fontWeight: 700,
            color: rd.color, letterSpacing: '0.03em',
            textShadow: `0 0 8px ${rd.color}, 0 0 2px #000`,
            lineHeight: 1.2,
          }}>
            {rd.label}
          </div>
          <div style={{
            fontSize: 8, fontFamily: 'monospace',
            color: actPct > 40 ? '#ffffff' : '#64748b',
            marginTop: 1,
          }}>
            {actPct}%
          </div>
        </div>
      </Html>
    </mesh>
  );
}

function RegionOrbs({ ns, refs }: { ns: NeuralState; refs: SpikeRefs }) {
  return (
    <group>
      {REGION_DEFS.map((rd) => (
        <RegionOrb key={rd.key} rd={rd} ns={ns} spikesRef={refs.spikes} />
      ))}
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 3. SYNAPSE ARCS — connection beams between regions
// ─────────────────────────────────────────────────────────────────────────
function SynapseArcs({ ns }: { ns: NeuralState }) {
  const groupRef = useRef<THREE.Group>(null!);

  // Stable arc geometry from top_routes
  const arcs = useMemo(() => {
    const routes = (ns.top_routes ?? []) as any[];
    return routes.slice(0, 8).map((r: any) => {
      const srcKey = r.src_region ?? r.src ?? '';
      const dstKey = r.dst_region ?? r.dst ?? '';
      const srcDef = REGION_DEFS.find((d) => d.key === srcKey);
      const dstDef = REGION_DEFS.find((d) => d.key === dstKey);
      if (!srcDef || !dstDef) return null;
      const s: [number, number, number] = [
        srcDef.pos[0] * SPHERE_R * 0.88,
        srcDef.pos[1] * SPHERE_R * 0.88,
        srcDef.pos[2] * SPHERE_R * 0.88,
      ];
      const d: [number, number, number] = [
        dstDef.pos[0] * SPHERE_R * 0.88,
        dstDef.pos[1] * SPHERE_R * 0.88,
        dstDef.pos[2] * SPHERE_R * 0.88,
      ];
      const mid: [number, number, number] = [
        (s[0] + d[0]) * 0.5,
        (s[1] + d[1]) * 0.5 + 0.8,
        (s[2] + d[2]) * 0.5,
      ];
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(...s),
        new THREE.Vector3(...mid),
        new THREE.Vector3(...d),
      );
      const pts = curve.getPoints(48);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const w = Math.min(1, (r.weight ?? 0.3) * 2.5);
      const col = new THREE.Color(srcDef.color);
      const mat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: w * 0.5 });
      return { geo, mat, w, srcKey };
    }).filter(Boolean);
  }, [ns.top_routes]);

  // Animate arc opacity (pulse when active)
  useFrame((state) => {
    if (!groupRef.current) return;
    const elapsed = state.clock.getElapsedTime();
    groupRef.current.children.forEach((child, i) => {
      const line = child as THREE.Line;
      const arc  = arcs[i];
      if (!arc) return;
      const act = getRegionAct((ns.regions ?? {}) as Record<string,number>, arc.srcKey);
      const base = arc.w * 0.35 + act * 0.4;
      const pulse = Math.sin(elapsed * 2.5 + i * 1.2) * 0.15;
      (line.material as THREE.LineBasicMaterial).opacity = Math.max(0.04, Math.min(0.9, base + pulse));
    });
  });

  return (
    <group ref={groupRef}>
      {arcs.map((arc, i) => arc && (
        <primitive key={i} object={new THREE.Line(arc.geo, arc.mat)} />
      ))}
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 4. THOUGHT BEAM — fires a travelling pulse when ThoughtGen picks a winner
// ─────────────────────────────────────────────────────────────────────────
function ThoughtBeam({ ns, thinking }: { ns: NeuralState; thinking: boolean }) {
  const groupRef    = useRef<THREE.Group>(null!);
  const beamMatRef  = useRef<THREE.LineBasicMaterial | null>(null);
  const beamGeoRef  = useRef<THREE.BufferGeometry | null>(null);
  const beamActive  = useRef(false);
  const beamT       = useRef(0);
  const prevThink   = useRef(false);
  // When thinking turns ON, spawn a beam from language → planning
  useEffect(() => {
    if (thinking && !prevThink.current) {
      // Pick the two most active regions as src→dst
      const regions = (ns.regions ?? {}) as Record<string,number>;
      const sorted  = REGION_DEFS
        .map((rd) => ({ rd, act: getRegionAct(regions, rd.key) }))
        .sort((a, b) => b.act - a.act);
      const src = sorted[0]?.rd ?? REGION_DEFS[8]; // language fallback
      const dst = sorted[1]?.rd ?? REGION_DEFS[0]; // planning fallback

      const sPos = src.pos.map((v) => v * SPHERE_R * 0.88) as [number,number,number];
      const dPos = dst.pos.map((v) => v * SPHERE_R * 0.88) as [number,number,number];
      const mid: [number,number,number] = [
        (sPos[0]+dPos[0])*0.5, (sPos[1]+dPos[1])*0.5+1.0, (sPos[2]+dPos[2])*0.5,
      ];
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(...sPos), new THREE.Vector3(...mid), new THREE.Vector3(...dPos),
      );
      const pts = curve.getPoints(80);
      if (beamGeoRef.current) beamGeoRef.current.dispose();
      const geo  = new THREE.BufferGeometry().setFromPoints(pts);
      beamGeoRef.current = geo;
      if (!beamMatRef.current) {
        beamMatRef.current = new THREE.LineBasicMaterial({
          color: new THREE.Color(src.color),
          transparent: true,
          opacity: 0,
        });
      } else {
        beamMatRef.current.color.set(src.color);
      }
      beamActive.current = true;
      beamT.current = 0;

      // Rebuild child
      if (groupRef.current) {
        while (groupRef.current.children.length) groupRef.current.remove(groupRef.current.children[0]);
        const line = new THREE.Line(geo, beamMatRef.current);
        groupRef.current.add(line);
      }
    }
    prevThink.current = thinking;
  }, [thinking, ns.regions]);

  useFrame((_, delta) => {
    if (!beamActive.current || !beamMatRef.current) return;
    beamT.current += delta;
    // Fade in fast, hold, then fade out
    const life = beamT.current;
    let opacity = 0;
    if (life < 0.3)      opacity = life / 0.3;         // fade in
    else if (life < 1.2) opacity = 1.0;                // hold
    else if (life < 1.8) opacity = 1 - (life - 1.2) / 0.6; // fade out
    else                 { beamActive.current = false; opacity = 0; }
    beamMatRef.current.opacity = opacity * 0.85;
  });

  return <group ref={groupRef} />;
}

// ─────────────────────────────────────────────────────────────────────────
// 5. LIGHTS
// ─────────────────────────────────────────────────────────────────────────
function DynamicLights({ ns }: { ns: NeuralState }) {
  const ambRef  = useRef<THREE.AmbientLight>(null!);
  const pt1Ref  = useRef<THREE.PointLight>(null!);
  const pt2Ref  = useRef<THREE.PointLight>(null!);
  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    const emo    = ((ns.emotion as any)?.current ?? (ns.emotion as any)?.emotion ?? 'neutral').toLowerCase();
    const emoCol = EMO_COLORS[emo] ?? '#475569';
    const nm     = (ns.neuromod as Record<string,number>) ?? {};
    const da     = nm['dopamine']  ?? 0.3;
    const ser    = nm['serotonin'] ?? 0.3;
    const surp   = ns.prediction_surprise ?? 0;

    ambRef.current?.color.lerp(new THREE.Color(emoCol), 0.025);
    if (ambRef.current) ambRef.current.intensity = 0.22 + ser * 0.12;

    if (pt1Ref.current) {
      const surpFlash = surp > 0.4 ? Math.exp(-((t.current % 1) * 5)) * 2 : 0;
      pt1Ref.current.intensity = 1.0 + da * 1.4 + surpFlash;
      pt1Ref.current.color.lerp(new THREE.Color(emoCol), 0.02);
    }
    if (pt2Ref.current) {
      pt2Ref.current.intensity = 0.5 + Math.sin(t.current * 0.4) * 0.1;
    }
  });

  return (
    <>
      <ambientLight ref={ambRef} intensity={0.25} color="#3344aa" />
      <pointLight ref={pt1Ref} position={[5, 5, 5]}   intensity={1.4} color="#6366f1" />
      <pointLight ref={pt2Ref} position={[-4,-3,-4]}  intensity={0.7} color="#22d3ee" />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 6. SHELL — outer glass + wireframe
// ─────────────────────────────────────────────────────────────────────────
function Shell({ ns }: { ns: NeuralState }) {
  const glassRef = useRef<THREE.Mesh>(null!);
  const wireRef  = useRef<THREE.Mesh>(null!);
  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    const conf     = (ns.cognitive_state as any)?.confidence ?? 0.5;
    const urgency  = (ns.cognitive_state as any)?.urgency    ?? 0;
    const conflict = (ns.conflict as any)?.score ?? 0;
    if (glassRef.current)
      (glassRef.current.material as THREE.MeshStandardMaterial).opacity = 0.025 + urgency * 0.03 + conflict * 0.025;
    if (wireRef.current)
      (wireRef.current.material as THREE.MeshBasicMaterial).opacity = 0.10 + conf * 0.14 + Math.sin(t.current * 0.4) * 0.03;
  });
  return (
    <>
      <mesh ref={glassRef}>
        <sphereGeometry args={[SPHERE_R + 0.06, 48, 48]} />
        <meshStandardMaterial color="#6366f1" transparent opacity={0.03} side={THREE.BackSide} />
      </mesh>
      <mesh ref={wireRef}>
        <sphereGeometry args={[SPHERE_R + 0.07, 16, 12]} />
        <meshBasicMaterial color="#1e1b4b" wireframe transparent opacity={0.12} />
      </mesh>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 7. THINKING RING
// ─────────────────────────────────────────────────────────────────────────
function ThinkingRing() {
  const groupRef = useRef<THREE.Group>(null!);
  const thinking = useAxonStore((s) => s.thinking);
  const thinkRef = useRef(false);
  const alpha    = useRef(0);
  useEffect(() => { thinkRef.current = thinking; }, [thinking]);

  const ringGeo = useMemo(() => new THREE.TorusGeometry(SPHERE_R + 0.28, 0.014, 8, 80), []);
  const ringMat = useMemo(() => new THREE.MeshBasicMaterial({
    color: '#6366f1', transparent: true, opacity: 0,
  }), []);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    alpha.current += ((thinkRef.current ? 0.75 : 0) - alpha.current) * delta * 3;
    ringMat.opacity = alpha.current;
    ringMat.color.set(thinkRef.current ? '#a5b4fc' : '#6366f1');
    groupRef.current.rotation.x += delta * 0.9;
    groupRef.current.rotation.z += delta * 0.55;
  });

  return <group ref={groupRef}><mesh geometry={ringGeo} material={ringMat} /></group>;
}

// ─────────────────────────────────────────────────────────────────────────
// 8. SCENE
// ─────────────────────────────────────────────────────────────────────────
function Scene() {
  const ns            = useAxonStore((s) => s.neuralState);
  const regionSpikes  = useAxonStore((s) => s.regionSpikes);
  const hebbianEvents = useAxonStore((s) => s.hebbianEvents);
  const thinking      = useAxonStore((s) => s.thinking);
  const { clock }     = useThree();

  const spikes       = useRef<Record<string, number>>({});
  const hebbianFlash = useRef<{ src: string; dst: string; t: number } | null>(null);

  // Inject spikes
  const lastSpikeLen = useRef(0);
  useEffect(() => {
    if (regionSpikes.length > lastSpikeLen.current) {
      const latest = regionSpikes[0];
      if (latest?.region) {
        spikes.current[latest.region] = Math.min(1, (latest.activation ?? 0.6) * 1.6);
      }
      lastSpikeLen.current = regionSpikes.length;
    }
  }, [regionSpikes]);

  // Inject hebbian
  const lastHebLen = useRef(0);
  useEffect(() => {
    if (hebbianEvents.length > lastHebLen.current) {
      const ev = hebbianEvents[0];
      if (ev) hebbianFlash.current = { src: ev.src ?? ev.a ?? '', dst: ev.dst ?? ev.b ?? '', t: clock.getElapsedTime() };
      lastHebLen.current = hebbianEvents.length;
    }
  }, [hebbianEvents, clock]);

  // Decay spikes
  useFrame((_, delta) => {
    Object.keys(spikes.current).forEach((rk) => {
      spikes.current[rk] = Math.max(0, spikes.current[rk] - delta * 2.8);
    });
  });

  const refs = { spikes };

  return (
    <>
      <DynamicLights ns={ns} />
      <NeuronParticles ns={ns} refs={refs} />
      <SynapseArcs    ns={ns} />
      <RegionOrbs     ns={ns} refs={refs} />
      <ThoughtBeam    ns={ns} thinking={thinking} />
      <Shell          ns={ns} />
      <ThinkingRing />
      <OrbitControls enablePan={false} minDistance={3.5} maxDistance={10} autoRotate autoRotateSpeed={0.25} />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// 9. ROOT
// ─────────────────────────────────────────────────────────────────────────
export default function BrainCanvas() {
  return (
    <div style={{ width: '100%', height: '100%', background: '#020205' }}>
      <Canvas
        camera={{ position: [0, 0, 5.8], fov: 52 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#020205']} />
        <Scene />
      </Canvas>
    </div>
  );
}

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  HsaaKnowledgeEngine,
  textToHypervector512,
  normalizedHammingSim512,
  runTrajectoryForecastExperiment,
  runDeterminismAndSimplexExperiment,
  runFormulaAndSystem2Experiment,
  runCodeSynthesisExperiment,
  evaluateAllFiveBlindSpotPatches,
} from './hsaa-engine';

test('VSA 512-bit hypervectors obey 0.40 resonance gate and orthogonal noise separation', () => {
  const vA = textToHypervector512('Модуль_Вега');
  const vB = textToHypervector512('Случайный_Шум_991');
  const selfSim = normalizedHammingSim512(vA, vA);
  const crossSim = normalizedHammingSim512(vA, vB);

  assert.equal(selfSim, 1.0);
  assert.ok(Math.abs(crossSim) < 0.25, `Expected orthogonal similarity < 0.25, got ${crossSim}`);
});

test('HsaaKnowledgeEngine supports exact lookup, typo SimHash recall, live override, and 2-hop deduction', () => {
  const kb = new HsaaKnowledgeEngine();
  kb.storeFact('Модуль_Вега', 'использует_архитектуру', 'Оптический_Тороид');
  kb.storeFact('Оптический_Тороид', 'работает_на_частоте', '420_ТГц');

  // 1. Exact lookup
  const exact = kb.queryRelation('Модуль_Вега', 'использует_архитектуру');
  assert.equal(exact.found, true);
  assert.equal(exact.value, 'Оптический_Тороид');
  assert.equal(exact.matchType, 'EXACT_DIRECT');

  // 2. Single-character typo recall via 512-bit trigram SimHash
  const typo = kb.queryRelation('Модуль_Вегa', 'использует_архитектуру');
  assert.equal(typo.found, true);
  assert.equal(typo.value, 'Оптический_Тороид');

  // 3. 2-Hop transitive deduction
  const hop2 = kb.queryRelation('Модуль_Вега', 'работает_на_частоте', true);
  assert.equal(hop2.found, true);
  assert.equal(hop2.value, '420_ТГц');
  assert.equal(hop2.matchType, 'TRANSITIVE_2HOP');

  // 4. Live fact mutation / override + conflict check
  kb.storeFact('Модуль_Вега', 'использует_архитектуру', 'Квантовый_Симплекс_v2');
  const conflictCheck = kb.checkConflict('Модуль_Вега', 'использует_архитектуру', 'Оптический_Тороид');
  assert.equal(conflictCheck.conflict, true);
  assert.equal(conflictCheck.storedObj, 'Квантовый_Симплекс_v2');
});

test('Trajectory forecasting trains RFF via SGD and separates in-basis vs out-of-basis extrapolation', () => {
  const ballistic = runTrajectoryForecastExperiment('ballistic', undefined, 60, 0.30);
  assert.ok(ballistic.rffFinalMse < ballistic.rffInitialMse, 'RFF SGD training must reduce MSE');
  assert.equal(ballistic.usedPiecewiseFallback, false);
  assert.ok(ballistic.interpolationMse.vsaHybrid < 0.05);
  assert.ok(ballistic.extrapolationMse.analytical5 < 1e-4);

  const outOfBasis = runTrajectoryForecastExperiment('damped_chirp', undefined, 60, 0.30, false);
  assert.equal(outOfBasis.usedPiecewiseFallback, true);
  assert.ok(outOfBasis.analyticalFitMse >= 1e-3);
});

test('Determinism & Simplex Gate: bitwise deterministic at T=0 and 0% contradiction leakage at T=1.2', () => {
  const kb = new HsaaKnowledgeEngine();
  const detZero = runDeterminismAndSimplexExperiment({
    temperature: 0.0,
    inputNoiseSigma: 0.0,
    numTrials: 12,
    tt16AdaptSteps: 10,
    knowledgeEngine: kb,
  });

  assert.equal(detZero.bitwiseIdenticalRatePct, 100);
  assert.equal(detZero.shannonEntropyBits, 0);
  assert.equal(detZero.dualRailAvgBitFlips, 0);
  assert.equal(detZero.contradictionLeakageRatePct, 0);
  assert.ok(detZero.tt16LossAfter < detZero.tt16LossBefore);

  const detStochastic = runDeterminismAndSimplexExperiment({
    temperature: 1.2,
    inputNoiseSigma: 0.08,
    numTrials: 24,
    tt16AdaptSteps: 10,
    knowledgeEngine: kb,
  });

  assert.ok(detStochastic.shannonEntropyBits > 0);
  assert.ok(detStochastic.dualRailAvgBitFlips > 0);
  assert.equal(detStochastic.contradictionLeakageRatePct, 0);
});

test('Symbolic formula discovery recovers exact basis-5 coefficients and synthesizes System-2 hypotheses', () => {
  const kb = new HsaaKnowledgeEngine();
  kb.storeFact('Модуль_Вега', 'использует_архитектуру', 'Оптический_Тороид');
  kb.storeFact('Оптический_Тороид', 'требует_охлаждения', 'Гелиевый_Контур_4К');
  kb.storeFact('Ядро_Гелиос', 'использует_архитектуру', 'Плазменный_Тороид');

  const harmonic = runFormulaAndSystem2Experiment('harmonic_exact', kb, 'Модуль_Вега');
  assert.equal(harmonic.basis5RecognizedExact, true);
  assert.equal(harmonic.fallbackTriggered, false);
  assert.ok(Math.abs(harmonic.basis5Coeffs[0] - 1.0) < 1e-2);
  assert.ok(Math.abs(harmonic.basis5Coeffs[3] - 1.8) < 1e-2);
  assert.ok(harmonic.system2Hypotheses.length >= 1);
});

test('Code synthesis performs zero-shot parameter renaming via <reg_0>..<reg_7> and validates AST', () => {
  const res = runCodeSynthesisExperiment(
    'SCOPE_REGISTER_INDUCTION',
    `int clamp_val(int x, int lo, int hi) {\n    if (x < lo) return lo;\n    if (x > hi) return hi;\n    return x;\n}`,
    `int clamp_val(int velocity_in, int min_limit, int max_limit)`
  );

  assert.equal(res.astBalancedValid, true);
  assert.equal(res.registerBindings.length, 3);
  assert.ok(res.generatedCode.includes('velocity_in'));
  assert.ok(res.generatedCode.includes('min_limit'));
  assert.ok(res.generatedCode.includes('max_limit'));
  assert.ok(!/\blo\b/.test(res.generatedCode));
});

test('HSAA v2.1 Patched Mode resolves all 5 blind spots empirically', () => {
  const kb = new HsaaKnowledgeEngine();
  kb.storeFact('Модуль_Вега', 'использует_архитектуру', 'Оптический_Тороид');
  kb.storeFact('Оптический_Тороид', 'требует_охлаждения', 'Гелиевый_Контур_4К');
  kb.storeFact('Ядро_Гелиос', 'питается_от', 'Квантовый_Резонатор');
  // Duplicate overwrite to verify deduplication in Zone 1
  kb.storeFact('Модуль_Вега', 'использует_архитектуру', 'Оптический_Тороид');

  // Zone 1: Patched page counter > 59 and deduplicated rules
  const pStats = kb.getPatchedStats();
  assert.equal(pStats.originalActivePages, 59);
  assert.ok(pStats.patchedActivePages > 59);
  assert.ok(pStats.patchedDeduplicatedRules < pStats.originalActiveRules);

  // Zone 2: Patched trajectory forecast on damped_chirp achieves MSE < 1e-5 without fallback
  const patchedChirp = runTrajectoryForecastExperiment('damped_chirp', undefined, 60, 0.30, true);
  assert.equal(patchedChirp.usedPiecewiseFallback, false);
  assert.ok(patchedChirp.analyticalFitMse < 1e-6);
  assert.ok(patchedChirp.extrapolationMse.analytical5 < 1e-5);

  // Zone 2 & 3: Patched formula discovery + System-2 manifold decoding (0 lexical concat artifacts)
  const patchedFormula = runFormulaAndSystem2Experiment(
    'exponential_decay_blindspot',
    kb,
    'Модуль_Вега',
    true
  );
  assert.equal(patchedFormula.basis5RecognizedExact, true);
  assert.equal(patchedFormula.fallbackTriggered, false);
  assert.ok(patchedFormula.heldOutTestMse < 1e-5);
  assert.ok(
    patchedFormula.system2Hypotheses.every((h) => !h.isLexicalConcatArtifact),
    'Patched System-2 must not emit objA + "_" + sB lexical concatenation artifacts'
  );

  // Verify all 5 zones in evaluateAllFiveBlindSpotPatches
  const patches = evaluateAllFiveBlindSpotPatches(kb);
  assert.equal(patches.length, 5);
  assert.ok(patches.every((p) => p.canBeFixed));
});

test('HSAA v3.0 Unified End-to-End Pipeline fuses VSA, DeltaNet, Spectral Head, and Code Synthesis on a single latent bus', async () => {
  const { runUnifiedHsaaV3Pipeline } = await import('./hsaa-engine');
  const kb = new HsaaKnowledgeEngine();
  kb.storeFact('Модуль_Вега', 'требует_охлаждения', 'Криогенный_Контур_4К');
  const res = runUnifiedHsaaV3Pipeline('cryo_plasma_stabilizer', kb, 24);

  assert.equal(res.stages.length, 4);
  assert.equal(res.latentBusPreview.length, 32);
  assert.equal(res.evalSeries.length, 16);
  assert.ok(res.jointLossFinal < res.jointLossInitial, 'Joint DeltaNet + TT16 loss must decrease');
  assert.ok(res.formulaFitMse < 1e-6, 'Neural Spectral Head must recover exact law from shared state');
  assert.ok(res.extrapolationMse < 1e-5, 'Extrapolation MSE must be near zero');
  assert.equal(res.astVerified, true);
  assert.equal(res.cegisVerified, true);
  assert.ok(res.synthesizedUnifiedCpp.includes('stabilize_plasma_field_v3'));

  const customRes = runUnifiedHsaaV3Pipeline('custom_multimodal', kb, 28, {
    querySubj: 'Модуль_Вега',
    queryRel: 'требует_охлаждения',
    fnName: 'custom_plasma_servo',
    paramNames: ['t_step', 'min_pwm', 'max_pwm'],
    gamma: 0.6,
    omega: 2.4,
    cubicCoeff: 0.04,
  });
  assert.ok(customRes.recalledFactChain[0].includes('Криогенный_Контур_4К'));
  assert.ok(customRes.synthesizedUnifiedCpp.includes('custom_plasma_servo'));
  assert.ok(customRes.synthesizedUnifiedCpp.includes('min_pwm'));
  assert.ok(customRes.formulaFitMse < 1e-6);
  assert.equal(customRes.selfAuditChecks.length, 6);
  assert.ok(customRes.hopfieldEnergyDelta < 0, 'Hopfield Lyapunov delta must be strictly negative');
});




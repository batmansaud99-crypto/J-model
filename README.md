# ============================================================
# Unified J Framework — تحليل SPARC الحقيقي الكامل
# Author: Saud Yasser Matar Al-Azmi (2026)
# ============================================================
# تعليمات:
# 1. ارفع ملف Rotmod_LTG.zip في Colab (أيقونة الملفات على اليسار)
# 2. انسخ هذا الكود كامل في خلية جديدة
# 3. اضغط Shift+Enter
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution
import zipfile, os, io, warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  Unified J Framework — SPARC Real Data Analysis")
print("  Saud Yasser Matar Al-Azmi (2026)")
print("="*60)

# ============================================================
# 1. قراءة بيانات SPARC الحقيقية
# ============================================================

def load_sparc_data(zip_path='Rotmod_LTG.zip', min_points=6):
    """قراءة كل مجرات SPARC من الملف الحقيقي"""
    
    galaxies = {}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('_rotmod.dat')]
        print(f"\n📂 وجدت {len(files)} مجرة في SPARC")
        
        for fname in files:
            name = fname.replace('_rotmod.dat', '')
            
            with z.open(fname) as f:
                lines = f.read().decode('utf-8').strip().split('\n')
            
            # استخراج المسافة
            distance = None
            for line in lines:
                if 'Distance' in line:
                    try:
                        distance = float(line.split('=')[1].split('Mpc')[0].strip())
                    except:
                        pass
            
            # قراءة البيانات
            data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
            
            rows = []
            for line in data_lines:
                try:
                    vals = [float(x) for x in line.split()]
                    if len(vals) >= 6:
                        rows.append(vals)
                except:
                    pass
            
            if len(rows) < min_points:
                continue
            
            arr = np.array(rows)
            r     = arr[:, 0]  # kpc
            v_obs = arr[:, 1]  # km/s
            v_err = arr[:, 2]  # km/s
            v_gas = arr[:, 3]  # km/s
            v_disk= arr[:, 4]  # km/s
            v_bul = arr[:, 5]  # km/s
            
            # تصفية القيم الغريبة
            mask = (v_obs > 0) & (v_err > 0) & (r > 0)
            if mask.sum() < min_points:
                continue
            
            r, v_obs, v_err = r[mask], v_obs[mask], v_err[mask]
            v_gas, v_disk, v_bul = v_gas[mask], v_disk[mask], v_bul[mask]
            
            # السرعة البارونية الكاملة
            v_bar = np.sqrt(np.abs(v_gas)**2 + np.abs(v_disk)**2 + np.abs(v_bul)**2)
            
            galaxies[name] = {
                'r': r,
                'v_obs': v_obs,
                'v_err': v_err,
                'v_bar': v_bar,
                'distance': distance
            }
    
    print(f"✅ تم تحميل {len(galaxies)} مجرة صالحة للتحليل")
    return galaxies

# ============================================================
# 2. نموذج J المحسّن
# ============================================================

def v_J_model(r, v_bar_arr, eta, r_scale, beta):
    """
    معادلة سرعة نموذج J الكاملة:
    v²(r) = v²_bar(r) + η · dI/dr
    
    دالة J المحسّنة مع معامل beta للمرونة الشكلية
    """
    v_bar = np.interp(r, np.linspace(r.min(), r.max(), len(v_bar_arr)), v_bar_arr)
    
    # دالة الكثافة المعلوماتية المحسّنة
    x = r / r_scale
    dI_dr = x * np.exp(-0.5 * x**beta)
    
    # تطبيع
    max_val = dI_dr.max()
    if max_val > 0:
        dI_dr = dI_dr / max_val
    
    v2 = v_bar**2 + eta * dI_dr
    return np.sqrt(np.maximum(v2, 1e-10))

def v_MOND(r, v_bar_arr):
    """نموذج MOND للمقارنة"""
    a0 = 1.2e-10  # m/s²
    # تحويل للوحدات الفلكية: 1 kpc = 3.086e19 m
    a0_astro = a0 * (3.086e19) / 1e6  # (km/s)²/kpc
    
    v_bar = np.interp(r, np.linspace(r.min(), r.max(), len(v_bar_arr)), v_bar_arr)
    
    # تسارع بارونية
    v_bar_safe = np.maximum(v_bar, 1e-3)
    a_bar = v_bar_safe**2 / r
    
    # دالة انتقال MOND
    mu = a_bar / np.sqrt(a_bar**2 + a0_astro**2)
    mu = np.maximum(mu, 1e-10)
    
    v2_MOND = v_bar**2 / mu
    return np.sqrt(np.maximum(v2_MOND, 1e-10))

# ============================================================
# 3. تحسين المعاملات
# ============================================================

def fit_galaxy(name, data):
    """تطبيق نموذج J على مجرة وإيجاد أفضل معاملات"""
    
    r     = data['r']
    v_obs = data['v_obs']
    v_err = data['v_err']
    v_bar = data['v_bar']
    
    r_max = r.max()
    
    def cost(params):
        eta, r_scale, beta = params
        try:
            v_pred = v_J_model(r, v_bar, eta, r_scale, beta)
            chi2 = np.sum(((v_obs - v_pred) / v_err)**2)
            return chi2
        except:
            return 1e12
    
    # حدود واقعية مبنية على خصائص المجرة
    v_max = v_obs.max()
    bounds = [
        (0, v_max**2 * r_max * 2),  # eta
        (r_max * 0.01, r_max * 3),   # r_scale
        (0.3, 4.0)                    # beta
    ]
    
    try:
        res = differential_evolution(
            cost, bounds,
            seed=42,
            maxiter=2000,
            tol=1e-10,
            popsize=25,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True
        )
        
        eta, r_scale, beta = res.x
        v_pred = v_J_model(r, v_bar, eta, r_scale, beta)
        
        n_params = 3
        dof = max(len(r) - n_params, 1)
        chi2 = res.fun / dof
        rms = np.sqrt(np.mean((v_obs - v_pred)**2))
        
        # MOND للمقارنة
        v_mond = v_MOND(r, v_bar)
        rms_mond = np.sqrt(np.mean((v_obs - v_mond)**2))
        chi2_mond = np.sum(((v_obs - v_mond) / v_err)**2) / dof
        
        return {
            'name': name,
            'eta': eta,
            'r_scale': r_scale,
            'beta': beta,
            'chi2': chi2,
            'rms': rms,
            'chi2_mond': chi2_mond,
            'rms_mond': rms_mond,
            'v_pred': v_pred,
            'v_mond': v_mond,
            'success': True,
            'n_points': len(r)
        }
    
    except Exception as e:
        return {'name': name, 'success': False}

# ============================================================
# 4. تشغيل التحليل الكامل
# ============================================================

def run_analysis(galaxies, max_galaxies=None):
    """تحليل كل المجرات"""
    
    names = list(galaxies.keys())
    if max_galaxies:
        names = names[:max_galaxies]
    
    print(f"\n🔭 تحليل {len(names)} مجرة...")
    print("-"*55)
    print(f"{'المجرة':<15} {'η':>10} {'r_sc':>7} {'χ²J':>7} {'χ²M':>7} {'J≷M':>5}")
    print("-"*55)
    
    results = []
    
    for i, name in enumerate(names):
        result = fit_galaxy(name, galaxies[name])
        
        if result['success']:
            better = "✅" if result['chi2'] < result['chi2_mond'] else "⚠️"
            print(f"{name:<15} {result['eta']:>10.0f} {result['r_scale']:>7.1f} "
                  f"{result['chi2']:>7.2f} {result['chi2_mond']:>7.2f} {better:>5}")
            results.append(result)
        
        # طباعة تقدم كل 20 مجرة
        if (i+1) % 20 == 0:
            print(f"  ... {i+1}/{len(names)} مكتمل")
    
    return results

# ============================================================
# 5. الإحصاءات النهائية
# ============================================================

def compute_statistics(results):
    """إحصاءات شاملة"""
    
    good = [r for r in results if r['chi2'] < 10 and r['rms'] < 30]
    
    etas     = [r['eta'] for r in good]
    rscales  = [r['r_scale'] for r in good]
    betas    = [r['beta'] for r in good]
    chi2s    = [r['chi2'] for r in good]
    rmss     = [r['rms'] for r in good]
    chi2m    = [r['chi2_mond'] for r in good]
    
    J_wins = sum(1 for r in results if r['chi2'] < r['chi2_mond'])
    
    # حساب alpha
    G_astro = 4.302e-3  # (km/s)² pc / M_sun
    eta_med = np.median(etas)
    alpha = eta_med / (G_astro * 1e11 * 1000)
    
    stats = {
        'n_total': len(results),
        'n_good': len(good),
        'J_wins': J_wins,
        'eta_mean': np.mean(etas),
        'eta_std': np.std(etas),
        'eta_median': np.median(etas),
        'r_scale_mean': np.mean(rscales),
        'r_scale_std': np.std(rscales),
        'r_scale_median': np.median(rscales),
        'beta_mean': np.mean(betas),
        'chi2_mean': np.mean(chi2s),
        'chi2_mond_mean': np.mean(chi2m),
        'rms_mean': np.mean(rmss),
        'alpha': alpha
    }
    
    print("\n" + "="*60)
    print("📊 النتائج الإجمالية — Unified J Framework")
    print("="*60)
    print(f"\n  المجرات المحللة:     {stats['n_total']}")
    print(f"  المجرات الجيدة:      {stats['n_good']} (χ²<10, RMS<30)")
    print(f"  J أفضل من MOND:      {J_wins}/{len(results)} مجرة ({100*J_wins/len(results):.1f}%)")
    print(f"\n  ─── معاملات نموذج J ───")
    print(f"  η (متوسط)   = {stats['eta_mean']:.0f} ± {stats['eta_std']:.0f} (km/s)² kpc")
    print(f"  η (وسيط)    = {stats['eta_median']:.0f} (km/s)² kpc")
    print(f"  r_scale     = {stats['r_scale_median']:.1f} ± {stats['r_scale_std']:.1f} kpc")
    print(f"  β           = {stats['beta_mean']:.2f}")
    print(f"  α           ~ {stats['alpha']:.2e}")
    print(f"\n  ─── جودة التطابق ───")
    print(f"  χ²_J (متوسط)    = {stats['chi2_mean']:.2f}")
    print(f"  χ²_MOND (متوسط) = {stats['chi2_mond_mean']:.2f}")
    print(f"  RMS_J           = {stats['rms_mean']:.1f} km/s")
    print(f"\n  تحسن J على MOND: {stats['chi2_mond_mean']/stats['chi2_mean']:.1f}x أفضل")
    
    return stats

# ============================================================
# 6. الرسوم البيانية
# ============================================================

def plot_best_galaxies(galaxies, results, n=12):
    """رسم أفضل المجرات تطابقاً"""
    
    # ترتيب حسب أفضل χ²
    sorted_res = sorted(results, key=lambda x: x['chi2'])[:n]
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, rows*4))
    fig.patch.set_facecolor('#080818')
    
    for idx, res in enumerate(sorted_res):
        ax = fig.add_subplot(rows, cols, idx+1)
        ax.set_facecolor('#0d0d2b')
        
        name = res['name']
        d = galaxies[name]
        r = d['r']
        
        # بيانات مرصودة
        ax.errorbar(r, d['v_obs'], yerr=d['v_err'],
                   fmt='o', color='#00ffcc', markersize=4,
                   capsize=2, linewidth=1, label='SPARC', zorder=5)
        
        # بارونية فقط
        ax.plot(r, d['v_bar'], '--', color='#ff6b6b',
               linewidth=1.5, alpha=0.7, label='بارونية')
        
        # MOND
        ax.plot(r, res['v_mond'], ':', color='#ff9f43',
               linewidth=1.5, alpha=0.8, label='MOND')
        
        # نموذج J
        r_s = np.linspace(r.min(), r.max(), 300)
        v_s = v_J_model(r_s, d['v_bar'], res['eta'], res['r_scale'], res['beta'])
        ax.plot(r_s, v_s, '-', color='#ffd700', linewidth=2.5, label='J', zorder=4)
        ax.fill_between(r_s, v_s*0.95, v_s*1.05, color='#ffd700', alpha=0.1)
        
        ax.set_title(name, color='white', fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel('r (kpc)', color='#888', fontsize=7)
        ax.set_ylabel('v (km/s)', color='#888', fontsize=7)
        ax.tick_params(colors='#888', labelsize=7)
        
        if idx == 0:
            ax.legend(fontsize=6, facecolor='#1a1a3e', labelcolor='white',
                     loc='lower right', framealpha=0.8)
        
        info = f"χ²={res['chi2']:.2f}\nRMS={res['rms']:.1f}"
        ax.text(0.04, 0.96, info, transform=ax.transAxes,
               color='#00ffcc', fontsize=7, va='top',
               bbox=dict(facecolor='#080818', alpha=0.7, pad=2))
        
        for sp in ax.spines.values():
            sp.set_color('#222244')
    
    plt.suptitle('Unified J Framework — Best Fits (Real SPARC Data)\nSaud Yasser Matar Al-Azmi (2026)',
                color='white', fontsize=12, fontweight='bold', y=1.01)
    
    plt.tight_layout(pad=1.5)
    plt.savefig('J_BestFits.png', dpi=150, bbox_inches='tight', facecolor='#080818')
    plt.show()
    print("✅ حُفظ: J_BestFits.png")

def plot_statistics(results, stats):
    """رسم توزيع المعاملات والإحصاءات"""
    
    good = [r for r in results if r['chi2'] < 10 and r['rms'] < 30]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('#080818')
    
    plots = [
        ([r['eta'] for r in good], 'η (km/s)² kpc', '#ffd700', 'توزيع قوة J'),
        ([r['r_scale'] for r in good], 'r_scale (kpc)', '#00ffcc', 'توزيع النطاق المعلوماتي'),
        ([r['beta'] for r in good], 'β', '#ff9f43', 'توزيع معامل الشكل'),
        ([r['chi2'] for r in good], 'χ²_red', '#a29bfe', 'جودة تطابق J'),
        ([r['chi2_mond'] for r in good], 'χ²_MOND', '#ff6b6b', 'جودة تطابق MOND'),
        ([r['rms'] for r in good], 'RMS (km/s)', '#55efc4', 'توزيع RMS'),
    ]
    
    for ax, (vals, xlabel, color, title) in zip(axes.flatten(), plots):
        ax.set_facecolor('#0d0d2b')
        
        ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor='#333366')
        ax.axvline(np.median(vals), color='white', linestyle='--',
                  linewidth=1.5, label=f'وسيط: {np.median(vals):.2f}')
        ax.axvline(np.mean(vals), color='yellow', linestyle=':',
                  linewidth=1.5, label=f'متوسط: {np.mean(vals):.2f}')
        
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, color='#888', fontsize=8)
        ax.set_ylabel('عدد المجرات', color='#888', fontsize=8)
        ax.tick_params(colors='#888')
        ax.legend(fontsize=7, facecolor='#1a1a3e', labelcolor='white')
        
        for sp in ax.spines.values():
            sp.set_color('#222244')
    
    plt.suptitle(f'إحصاءات نموذج J — {stats["n_good"]} مجرة من SPARC\nSaud Yasser Matar Al-Azmi (2026)',
                color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('J_Statistics.png', dpi=150, bbox_inches='tight', facecolor='#080818')
    plt.show()
    print("✅ حُفظ: J_Statistics.png")

def plot_J_vs_MOND(results):
    """مقارنة مباشرة J vs MOND"""
    
    chi2_J    = [r['chi2'] for r in results if r['success']]
    chi2_MOND = [r['chi2_mond'] for r in results if r['success']]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('#080818')
    
    # Scatter: J vs MOND
    ax = axes[0]
    ax.set_facecolor('#0d0d2b')
    
    colors = ['#00ffcc' if j < m else '#ff6b6b' for j, m in zip(chi2_J, chi2_MOND)]
    ax.scatter(chi2_MOND, chi2_J, c=colors, alpha=0.6, s=20, edgecolors='none')
    
    lim = min(max(chi2_MOND), 50)
    ax.plot([0, lim], [0, lim], '--', color='white', linewidth=1, alpha=0.5, label='خط التساوي')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    
    J_wins = sum(1 for j, m in zip(chi2_J, chi2_MOND) if j < m)
    ax.set_title(f'J vs MOND\nJ أفضل في {J_wins}/{len(chi2_J)} مجرة ({100*J_wins/len(chi2_J):.1f}%)',
                color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel('χ² MOND', color='#888')
    ax.set_ylabel('χ² J', color='#888')
    ax.tick_params(colors='#888')
    ax.legend(fontsize=8, facecolor='#1a1a3e', labelcolor='white')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#00ffcc', label='J أفضل'),
                       Patch(facecolor='#ff6b6b', label='MOND أفضل')]
    ax.legend(handles=legend_elements, fontsize=8,
             facecolor='#1a1a3e', labelcolor='white')
    
    for sp in ax.spines.values():
        sp.set_color('#222244')
    
    # Bar: متوسط χ²
    ax2 = axes[1]
    ax2.set_facecolor('#0d0d2b')
    
    good_J    = [x for x in chi2_J if x < 20]
    good_MOND = [x for x in chi2_MOND if x < 20]
    
    bars = ax2.bar(['J Framework', 'MOND'],
                   [np.median(good_J), np.median(good_MOND)],
                   color=['#ffd700', '#ff6b6b'],
                   width=0.5, edgecolor='white', linewidth=0.5)
    
    ax2.axhline(1.0, color='#00ffcc', linestyle='--',
               linewidth=1.5, label='χ²=1 (مثالي)')
    
    for bar, val in zip(bars, [np.median(good_J), np.median(good_MOND)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', color='white', fontsize=11, fontweight='bold')
    
    ax2.set_title('وسيط χ² — J مقابل MOND', color='white', fontsize=10, fontweight='bold')
    ax2.set_ylabel('وسيط χ²_red', color='#888')
    ax2.tick_params(colors='#888')
    ax2.legend(fontsize=8, facecolor='#1a1a3e', labelcolor='white')
    
    for sp in ax2.spines.values():
        sp.set_color('#222244')
    
    plt.suptitle('Unified J Framework vs MOND — SPARC الكامل',
                color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('J_vs_MOND.png', dpi=150, bbox_inches='tight', facecolor='#080818')
    plt.show()
    print("✅ حُفظ: J_vs_MOND.png")

# ============================================================
# 7. التقرير النهائي للورقة
# ============================================================

def print_paper_report(stats, results):
    """تقرير جاهز للورقة البحثية"""
    
    print("\n" + "="*60)
    print("📄 التقرير الجاهز للورقة البحثية")
    print("="*60)
    
    J_wins = sum(1 for r in results if r['chi2'] < r['chi2_mond'])
    improvement = stats['chi2_mond_mean'] / stats['chi2_mean']
    
    print(f"""
  النتائج الرئيسية (من {stats['n_total']} مجرة SPARC حقيقية):

  المعاملات المقدرة:
  ──────────────────────────────────────────
  η (eta)    = {stats['eta_median']:.0f} (km/s)² kpc  [وسيط]
  r_scale    = {stats['r_scale_median']:.1f} kpc          [وسيط]  
  β (beta)   = {stats['beta_mean']:.2f}
  α (alpha)  ~ {stats['alpha']:.2e}

  جودة التطابق:
  ──────────────────────────────────────────
  χ²_red (J)    = {stats['chi2_mean']:.2f}
  χ²_red (MOND) = {stats['chi2_mond_mean']:.2f}
  RMS (J)       = {stats['rms_mean']:.1f} km/s

  المقارنة:
  ──────────────────────────────────────────
  J أفضل من MOND في {J_wins}/{stats['n_total']} مجرة
  تحسن J على MOND: {improvement:.1f}x
  
  الصور المحفوظة:
  ──────────────────────────────────────────
  📊 J_BestFits.png    — أفضل 12 تطابق
  📈 J_Statistics.png  — توزيع المعاملات
  ⚖️  J_vs_MOND.png    — مقارنة شاملة
    """)
    
    print("="*60)
    print("✅ الورقة جاهزة للتحديث بهذه النتائج")
    print("="*60)

# ============================================================
# 8. تشغيل كل شيء
# ============================================================

# تحميل البيانات
galaxies = load_sparc_data('Rotmod_LTG.zip')

# تحليل كل المجرات (اجعل max_galaxies=None للكل، أو رقم للاختبار)
results = run_analysis(galaxies, max_galaxies=None)

if results:
    # الإحصاءات
    stats = compute_statistics(results)
    
    # الرسوم البيانية
    plot_best_galaxies(galaxies, results, n=12)
    plot_statistics(results, stats)
    plot_J_vs_MOND(results)
    
    # التقرير النهائي
    print_paper_report(stats, results)
else:
    print("❌ لم تنجح أي عملية تحليل")

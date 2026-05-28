'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  Image,
  Radar,
  Layers,
  FileText,
  Sparkles,
  Upload,
  Settings,
  Brain,
} from 'lucide-react';

const navigation = [
  { name: '控制台', href: '/', icon: Home },
  { name: '图像分类', href: '/classification', icon: Image },
  { name: '目标检测', href: '/detection', icon: Radar },
  { name: '图像分割', href: '/segmentation', icon: Layers },
  { name: '自然语言', href: '/nlp', icon: FileText },
  { name: '图像生成', href: '/generation', icon: Sparkles },
  { name: '模型训练', href: '/training', icon: Brain },
  { name: '模型管理', href: '/models', icon: Upload },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 h-screen bg-[#1e293b] border-r border-slate-700 flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">PyTorch 工作室</h1>
            <p className="text-xs text-slate-400">深度学习平台</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`
                    flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all
                    ${
                      isActive
                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                        : 'text-slate-400 hover:bg-slate-700/50 hover:text-white'
                    }
                  `}
                >
                  <item.icon size={20} />
                  <span className="text-sm font-medium">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span>后端已连接</span>
        </div>
      </div>
    </aside>
  );
}

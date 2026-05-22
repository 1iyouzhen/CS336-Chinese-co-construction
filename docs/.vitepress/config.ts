import { defineConfig, type HeadConfig } from 'vitepress'
import { generateSidebar } from 'vitepress-sidebar'
import { withMermaid } from 'vitepress-plugin-mermaid'
import timeline from "vitepress-markdown-timeline"
import { writeFile } from 'fs/promises'
import { join } from 'path'

// ========== 一、站点配置（统一管理 SEO 和资源配置）==========
const SITE_CONFIG = {
  // 站点标题配置
  title: "Diy-LLM",  // 浏览器标签页标题、SEO 标题
  siteTitle: "Diy-LLM",  // 左上角导航栏标题（非SEO）
  description: "一座为中文学习者量身打造的'LLM炼丹工坊'",

  // SEO 配置
  url: 'https://datawhalechina.github.io/diy-llm/',  // 网站域名，示例: 'https://yourdomain.com'（留空则不生成 sitemap）
  keywords: 'LLM,大语言模型,教程,分词器,Transformer,推理,训练,评测',
  author: '',

  // 资源配置
  favicon: {
    href: '/favicon/datawhale.png',// 网站图标
    type: 'image/png'  // 支持: image/png, image/svg+xml, image/x-icon 等(tips:手动修改匹配)
  },
  logo: '/favicon/datawhale.png',// 左上角图标

  // robots.txt 排除目录（根据项目实际情况调整）
  robotsDisallow: [
    '/*assets/',     // 任意层级名为 assets 的文件夹（兼容主流爬虫的通配符）
    '/.vitepress/',  // VitePress 配置
  ],
}

/**
 * 将 Markdown 文件路径转换为网站 URL 路径
 * 兼容 rewrites 和国际化路由配置
 */
function getPageUrl(relativePath: string): string {
  return relativePath
    .replace(/\\/g, '/')              // Windows 路径 -> Unix 路径
    .replace(/\.md$/, '.html')        // .md -> .html
    .replace(/\/index\.html$/, '/')    // /index.html -> /
    .replace(/^index\.html$/, '')      // 根 index.html -> ''
    .replace(/^/, '/')                 // 确保以 / 开头
    .replace(/\/\/+/g, '/')            // 去除多余斜杠
}

// ========== 二、侧边栏自动化生成 ==========
const commonSidebarConfig = {
  useTitleFromFileHeading: true,
  useFolderTitleFromIndexFile: true,
  useFolderLinkFromIndexFile: true,
  hyphenToSpace: true,
  collapsed: true,
  excludePattern: ['public', 'assets', 'docs'],
  manualSortFileNameByPriority: [ // 手动排序文件名优先级
    '',
    '',
    ''
  ],
}

// 为侧边栏所有链接添加国际化路径前缀
const addPrefix = (items: any, prefix: string): any => {
  if (Array.isArray(items)) {
    return items.map(item => ({
      ...item,
      link: item.link ? prefix + item.link.replace(/^\//, '') : undefined,
      items: item.items ? addPrefix(item.items, prefix) : undefined
    }))
  }
  return items
}

const getSidebarOrder = (item: any): number => {
  const text = String(item?.text ?? '')
  const chapterMatch = text.match(/^第\s*(\d+)\s*章/)
  if (chapterMatch) {
    return Number(chapterMatch[1])
  }

  const linkMatch = String(item?.link ?? '').match(/chapter(\d+)/)
  if (linkMatch) {
    return Number(linkMatch[1])
  }

  return Number.POSITIVE_INFINITY
}

const sortSidebarItems = (items: any): any => {
  if (!Array.isArray(items)) {
    return items
  }

  return [...items]
    .sort((left, right) => getSidebarOrder(left) - getSidebarOrder(right))
    .map(item => ({
      ...item,
      items: item.items ? sortSidebarItems(item.items) : undefined,
    }))
}

// 生成侧边栏（支持国际化前缀）
const createSidebar = (root: string, prefix = '/') => {
  const sidebar = generateSidebar({ documentRootPath: root, ...commonSidebarConfig })
  const sortedSidebar = sortSidebarItems(sidebar)
  return prefix === '/' ? sortedSidebar : addPrefix(sortedSidebar, prefix)
}

const zhNav = [
  {
    text: '基础必学',
    link: '/chapter0/',
    items: [
      { text: '前言', link: '/chapter0/' },
      { text: '工具使用', link: '/chapter1/' },
      { text: '分词器', link: '/chapter2/' },
      { text: 'PyTorch与资源核算', link: '/chapter3/' },
      { text: '架构与细节', link: '/chapter4/' },
      { text: 'MOE', link: '/chapter5/' },
      { text: 'GPU优化', link: '/chapter6/' },
      { text: '数据工程', link: '/chapter11/' },
      { text: '训练流程', link: '/chapter13/' },
      { text: '评估与基准测试', link: '/chapter12/' },
    ],
  },
  {
    text: '进阶选修',
    link: '/chapter8/',
    items: [
      { text: '分布式训练', link: '/chapter8/' },
      { text: 'GPU高性能编程', link: '/chapter7/' },
      { text: 'Scaling Laws', link: '/chapter9/' },
      { text: '推理', link: '/chapter10/' },
      { text: 'RLVR', link: '/chapter14/' },
    ],
  },
  {
    text: '扩展章节',
    link: '/chapter15/',
    items: [
      { text: '第15章 扩展内容', link: '/chapter15/' },
    ],
  },
]

const fundamentalSidebar = [
  {
    text: '基础必学',
    collapsed: false,
    items: [
      { text: '前言', link: '/chapter0/' },
      { text: '第1章 工具使用', link: '/chapter1/' },
      { text: '第2章 分词器', link: '/chapter2/' },
      { text: '第3章 PyTorch 与资源核算', link: '/chapter3/' },
      { text: '第4章 语言模型架构与训练细节', link: '/chapter4/' },
      { text: '第5章 MOE', link: '/chapter5/' },
      { text: '第6章 GPU优化', link: '/chapter6/' },
      { text: '第11章 数据工程', link: '/chapter11/' },
      { text: '第13章 训练流程', link: '/chapter13/' },
      { text: '第12章 评估与基准测试', link: '/chapter12/' },
    ],
  },
]

const advancedSidebar = [
  {
    text: '进阶选修',
    collapsed: false,
    items: [
      { text: '第8章 分布式训练', link: '/chapter8/' },
      { text: '第7章 GPU高性能编程', link: '/chapter7/' },
      { text: '第9章 Scaling Laws', link: '/chapter9/' },
      { text: '第10章 推理', link: '/chapter10/' },
      { text: '第14章 RLVR', link: '/chapter14/' },
    ],
  },
]

const extensionSidebar = [
  {
    text: '扩展章节',
    collapsed: false,
    items: [
      { text: '第15章 扩展内容', link: '/chapter15/' },
    ],
  },
]

const zhSidebar = {
  '/chapter0/': fundamentalSidebar,
  '/chapter1/': fundamentalSidebar,
  '/chapter2/': fundamentalSidebar,
  '/chapter3/': fundamentalSidebar,
  '/chapter4/': fundamentalSidebar,
  '/chapter5/': fundamentalSidebar,
  '/chapter6/': fundamentalSidebar,
  '/chapter11/': fundamentalSidebar,
  '/chapter12/': fundamentalSidebar,
  '/chapter13/': fundamentalSidebar,
  '/chapter7/': advancedSidebar,
  '/chapter8/': advancedSidebar,
  '/chapter9/': advancedSidebar,
  '/chapter10/': advancedSidebar,
  '/chapter14/': advancedSidebar,
  '/chapter15/': extensionSidebar,
}

// ========== 三、VitePress 配置 ==========

// 注意：如果网站部署在根目录下（例如使用自定义域名 https://your-domain.com/），
// 请将下方的 '/diy-llm/' 改为 '/'
// 如果部署在 GitHub Pages 子路径（https://username.github.io/diy-llm/），则保持 '/diy-llm/'
const deployPath = '/diy-llm/'

// 根据环境动态设置 base 路径 (生产环境用 deployPath, 本地开发强制用 /)
const base = process.env.NODE_ENV === 'production' ? deployPath : '/'

export default withMermaid(defineConfig({

  // 部署时的路径前缀
  base,

  // 路由重写：将 en 目录映射到根路径,作为默认语言内容
  rewrites: {
    'zh/index.md': 'index.md',            // 中文首页映射到根路径
    'zh/:dir/:rest*': ':dir/:rest*',      // 中文内容映射到根路径
  },

  // 排除目录
  srcExclude: ['**/docs/**'],

  //主题配置（全局配置，会被 locales 中的配置继承）
  themeConfig: {
    logo: SITE_CONFIG.logo,//左上角logo
    siteTitle: SITE_CONFIG.siteTitle,//左上角标题

    socialLinks: [//外部链接图标配置    
      { icon: 'github', link: 'https://github.com/datawhalechina/diy-llm' },
    ],

    footer: { //底部版权信息配置
      message: '© 2026 Datawhale. All Rights Reserved.',
    },

    // 全局搜索配置&UI语言设置(英文en无需再次配置)
    search: {
      provider: 'local',
      options: {
        locales: {
          zh: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
          // 添加其他语言示例：
          // fr: { translations: { /* 法语翻译 */ } }
        }
      }
    }
  },


  // ========== 国际化页面配置 ==========
  locales: {
    // ---------- 中文配置 ----------
    root: {
      label: '中文',
      lang: 'zh-CN',
      title: SITE_CONFIG.title,
      description: SITE_CONFIG.description,
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outlineTitle: '页面导航',
        lastUpdatedText: '最后更新于',
        darkModeSwitchLabel: '主题',
        sidebarMenuLabel: '菜单',
        returnToTopLabel: '回到顶部',
        langMenuLabel: '多语言',

        docFooter: {
          prev: '上一页',
          next: '下一页'
        }
      }
    },

    // ---------- 英文配置 ----------
    en: {
      label: 'English',
      lang: 'en',
      title: SITE_CONFIG.title,
      description: SITE_CONFIG.description,
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Chapters', link: '/en/chapter1/' },
        ],
        sidebar: createSidebar('docs/en', '/en/'),
      }
    }
  },

  // ========== 四、默认设置 ==========

  // 1.显示最后更新时间
  lastUpdated: true,

  // 2. Markdown 增强配置
  markdown: {
    math: true,// 开启数学公式 ($$ E=mc^2 $$)
    lineNumbers: true, // 开启行号
    languageAlias: {   // 语言别名，消除 gitignore/env 警告
      'gitignore': 'ini',
      'env': 'properties'
    },
    config: (md) => {// 注册时间线插件
      md.use(timeline);
    },
  },

  // 3. Mermaid 配置
  mermaid: {// refer to mermaid config options
  },
  mermaidPlugin: {
    class: "mermaid my-class", // set additional css classes for parent container 
  },

  // 4. Vite 构建配置
  vite: {
    // SSR（服务端渲染）配置
    ssr: {
      noExternal: ['vitepress-plugin-mermaid', 'mermaid'],// 将这些包打包到输出中，解决 Mermaid 图表在 SSR 时的兼容性问题
    },
    // 优化mermaid配置
    optimizeDeps: {
      include: ['mermaid'],// 提前预构建 mermaid，提升开发环境性能
    },
    // 开发服务器配置
    server: {
      host: '0.0.0.0',      // 监听所有网络接口（可从局域网访问）
      port: 5173,           // 默认端口
      strictPort: false,    // 端口被占用时自动尝试下一个端口
      allowedHosts: true    // 允许所有主机访问（跳过 host 检查）
    }
  },

  // ========== 五、SEO 优化配置 ==========

  // 基础 head 标签（全局生效）
  head: [
    ['meta', { name: 'keywords', content: SITE_CONFIG.keywords }],
    ['meta', { name: 'author', content: SITE_CONFIG.author }],
    ['meta', { property: 'og:type', content: 'website' }],
    // 注意：head 中的资源路径不会自动添加 base，需要手动处理
    ['link', { rel: 'icon', href: base + SITE_CONFIG.favicon.href.replace(/^\//, ''), type: SITE_CONFIG.favicon.type }],
  ],

  // Sitemap 自动生成（需配置 SITE_CONFIG.url）
  ...(SITE_CONFIG.url ? { sitemap: { hostname: SITE_CONFIG.url } } : {}),

  // 动态生成每个页面的 SEO meta 标签（标题、描述、URL、社交分享图）
  transformHead: ({ pageData }) => {
    if (!SITE_CONFIG.url) return []

    const pageUrl = `${SITE_CONFIG.url}${getPageUrl(pageData.relativePath)}`
    const title = pageData.frontmatter.title || pageData.title
    const description = pageData.frontmatter.description || pageData.description
    const image = `${SITE_CONFIG.url}${base}${SITE_CONFIG.logo.replace(/^\//, '')}`

    return [
      // 规范链接（避免重复内容）
      ['link', { rel: 'canonical', href: pageUrl }],
      // Open Graph（Facebook、微信等）
      ['meta', { property: 'og:url', content: pageUrl }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { property: 'og:image', content: image }],
      // Twitter Card
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['meta', { name: 'twitter:image', content: image }],
    ] as HeadConfig[]
  },

  // 构建完成后自动生成 robots.txt
  buildEnd: async (siteConfig) => {
    const disallowRules = SITE_CONFIG.robotsDisallow
      .map(path => `Disallow: ${path}`)
      .join('\n')

    const robotsContent = [
      'User-agent: *',
      'Allow: /',
      '',
      '# 排除资源文件',
      disallowRules,
      '',
      ...(SITE_CONFIG.url ? [`Sitemap: ${SITE_CONFIG.url}/sitemap.xml`] : []),
    ].join('\n')

    await writeFile(join(siteConfig.outDir, 'robots.txt'), robotsContent, 'utf-8')
  },
}))

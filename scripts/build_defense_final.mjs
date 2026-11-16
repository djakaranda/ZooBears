import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES=modules;
const require=createRequire(path.join(modules,'package.json'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const skill='C:/Users/Макар/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const dir=path.join(root,'tmp/defense_final');await fs.mkdir(path.join(dir,'previews'),{recursive:true});
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const report=await read('reports/sprint_val_metrics.json');const m=report.official;
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#F7F8FA',ink:'#101C30',blue:'#244BEE',muted:'#617088',line:'#D8DFE9',dark:'#09111F',white:'#FFFFFF',mint:'#20C6AA'};
const all=[];const tableOwners=[],chartOwners=[];
function text(slide,value,x,y,w,h,size=26,color=C.ink,bold=false){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Arial',fontSize:size,color,bold,verticalAlignment:'top',wrap:true};s.text.insets=0;return s;
}
function slide(title,note,source='reports/SPRINT_REPORT.md',dark=false){
 const s=presentation.slides.add();s.background.fill=dark?C.dark:C.bg;
 all.push({title,note,source});
 if(title)text(s,title,64,46,1152,112,46,dark?C.white:C.ink,true);
 text(s,String(all.length).padStart(2,'0'),1172,674,44,22,16,dark?'#93AACB':C.muted);
 s.speakerNotes.text=note+'\n\nИсточники: '+source+'\nКорень репозитория: https://github.com/djakaranda/ZooBears';return s;
}
function footer(s,t){text(s,t,64,653,1080,43,17,C.muted);}
function table(s,values,x=64,y=182,w=1152,h=380,widths=null,size=25){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
 tableOwners.push(all.length);t.borders.assign({fill:C.line,width:.65,style:'solid'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=h/values.length;
  for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2===1?'#FFFFFF':C.bg);
   cell.text.style={typeface:'Arial',fontSize:size,color:r===0?C.white:C.ink,bold:r===0||c===0};
  }
 }
 return t;
}
const coverBytes=await fs.readFile(path.join(root,'docs/presentation_assets/zoobears_cover.png'));
const coverBlob=coverBytes.buffer.slice(coverBytes.byteOffset,coverBytes.byteOffset+coverBytes.byteLength);
function coverImage(s){s.images.add({blob:coverBlob,contentType:'image/png',alt:'Скульптурный медведь, символ команды ZooBears',position:{left:0,top:0,width:1280,height:720},fit:'cover'});}
// 01
{
 const s=slide('', '20 секунд. Мы ZooBears. Мы дообучили компактную модель для извлечения фактов из русских обращений в поддержку. На полном открытом val наша версия проходит оба порога заказчика по датам и текущему состоянию. Далее покажем все метрики, устройство решения и ограничения.', 'reports/SPRINT_REPORT.md. Иллюстрация: сгенерированная командная символика.',true);
 coverImage(s);text(s,'ZooBears',64,174,690,110,100,C.white,true);
 text(s,'Компактное извлечение фактов\nиз звонков поддержки',68,312,680,100,36,'#DBE8FC');
 text(s,'Qwen3-0.6B',68,474,560,50,40,'#79C8FF',true);
 text(s,'Innopolis Global Hack 2026\nМТС MWS CompactFactExtraction',68,605,720,58,22,'#AFC2DD');
 all[0].title='ZooBears';
}
// 02
{
 const s=slide('Задача: факты с датами и источником','30 секунд. Контакт-центр должен получать не пересказ разговора, а записи с объектом, состоянием, периодом действия и доказательством в реплике. На этом примере новая дата закрывает прежнее состояние заявки. Выход использует схему организаторов.','data/raw/schema.md; reports/defense/explicit_transition.json');
 text(s,'Разговор',64,180,500,42,28,C.blue,true);
 text(s,'«Заявка 5160 была принята\n31 марта».\n\n«С 4 апреля заявка 5160\nв работе».',64,246,550,260,32);
 text(s,'Факты',694,180,500,42,28,C.blue,true);
 text(s,'Заявка 5160\nСтатус: принята\nПериод: 31 марта – 4 апреля\nИсточник: реплика 2\n\nСтатус: в работе с 4 апреля\nИсточник: реплика 8',694,246,510,330,28);
 footer(s,'Поддерживаем состояния услуг, тарифы, заявки, действия и договорённости.');
}
// 03
{
 const s=slide('Архитектура экстрактора','40 секунд. Входом служат дата звонка и реплики с говорящими. Qwen выделяет компактные записи и ссылки на даты из текущего разговора. CPU-ранжировщик уточняет границы состояний. Календарные правила нормализуют даты, после чего код проверяет схему и восстанавливает цитаты. Вторую языковую модель на inference не используем.','src/compactfacts/qwen.py; src/compactfacts/scoped_dates.py; artifacts/qwen_sprint/inference_config.json');
 const stages=[['01','Транскрипт и дата звонка','Реплики клиента и оператора'],['02','Qwen3-0.6B с LoRA','Объект, атрибут, значение, полярность, ссылки на даты'],['03','CPU-модели и календарные правила','Начало, окончание, последовательность состояний'],['04','Проверка и JSON API','Цитата, исходная реплика, неопределённость даты']];
 stages.forEach((a,i)=>{const y=182+i*104;text(s,a[0],64,y,70,40,30,C.blue,true);text(s,a[1],158,y,1010,42,30,C.ink,true);text(s,a[2],158,y+45,1010,43,24,C.muted);});
}
// 04
{
 const s=slide('Данные и дообучение','40 секунд. В исходном train 3926 звонков. Мы разделили его по аккаунтам: 3334 fit и 592 dev. Пересечения аккаунтов нет. Последнее продолжение использовало 800 fit-разговоров, 640 новых синтетических примеров и 40 защищённых ASR-копий. Новый набор из 43 сценариев не вошёл в fit. Эпоха завершена за 37 минут. Step100 выбрали по полному dev, а не по одному token loss. Val уже использовался в прежнем анализе ошибок, поэтому независимую оценку даст скрытый тест.','reports/sprint_training_data_audit.json; reports/sprint_data_audit.json; reports/sprint_selection.json');
 [['3334','fit'],['592','dev'],['383','внешний val']].forEach((a,i)=>{text(s,a[0],64+i*398,182,360,100,80,C.blue,true);text(s,a[1],68+i*398,284,350,45,30);});
 text(s,'Продолжение LoRA',64,390,550,42,30,C.ink,true);
 text(s,'800 исходных + 640 синтетических звонков\n40 ASR-копий, 1545 обучающих окон',64,450,1100,88,28);
 text(s,'10,09 млн обучаемых параметров. Полная эпоха: 37 минут.',64,565,1135,46,28,C.blue,true);
 footer(s,'Разделение по аккаунтам. Checkpoint step100 выбираем на полном dev.');
}
// 05
{
 const s=slide('Все метрики качества на val','45 секунд. Здесь сравнение с официальным большим LLM-экстрактором организаторов, а не с нашей прежней моделью. Precision и recall заметно выше базы. Даты и текущее состояние немного ниже базы, но в пределах допуска. Grounding turn выше базы. Span у базы не рассчитан. Метрика span учитывает пересечение фрагментов среди сопоставленных фактов с правильной репликой.','data/raw/baseline_metrics.json; reports/sprint_val_metrics.json; data/raw/case_metrics.py');
 table(s,[['Метрика','LLM-база','ZooBears'],['Precision','33,48%','60,22%'],['Recall','38,05%','56,37%'],['Temporal','47,06%','45,89%'],['State','53,45%','52,97%'],['Grounding turn','78,35%','80,48%'],['Grounding span','нет оценки','96,99%']],64,173,1152,427,[620,266,266],25);
 footer(s,'Оценщик v2.1. Все 383 звонка. Span означает пересечение цитат, а не точные границы.');
}
// 06
{
 const s=slide('Оба порога качества пройдены','35 секунд. Допуск составляет два процентных пункта относительно базы. Для Temporal нужен минимум 45,06, у нас 45,89. Для State нужен 51,45, у нас 52,97. Ветка сокращения числа параметров обоснована сравнением с опубликованной моделью учителя: консервативная оценка студента не более 610 миллионов. Это не доказательство ускорения или денежной экономии.','reports/sprint_threshold_check.json; reports/TEACHER_PARAMETER_COMPARISON.md');
 const chart=s.charts.add('bar',{position:{left:56,top:180,width:790,height:395},categories:['Temporal','State'],series:[{name:'Минимальный порог',values:[.4506,.5145],fill:'#BBC7DA'},{name:'ZooBears',values:[.4589,.5297],fill:C.blue}],hasLegend:true,legend:{position:'bottom',textStyle:{typeface:'Arial',fontSize:22,fill:C.ink}},barOptions:{direction:'column',grouping:'clustered',gapWidth:95},yAxis:{min:0,max:.6,majorUnit:.2,numberFormatCode:'0%',textStyle:{typeface:'Arial',fontSize:20,fill:C.muted},majorGridlines:{fill:C.line,width:1}},xAxis:{textStyle:{typeface:'Arial',fontSize:25,fill:C.ink}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00%',textStyle:{typeface:'Arial',fontSize:23,fill:C.ink,bold:true}},chartFill:C.bg,plotAreaFill:C.bg});
 applyPresentationChartFont(chart,{fontFamily:'Arial'});chartOwners.push(all.length);
 text(s,'596 млн',908,210,310,84,59,C.blue,true);text(s,'базовых параметров',908,301,310,75,26);
 text(s,'0,5–1,5 млрд',908,416,310,54,34,C.ink,true);text(s,'диапазон менторов\nсоблюдён',908,478,310,85,25,C.muted);
 footer(s,'Параметрическая ветка ≥10× обоснована. Ускорение и экономию денег относительно LLM не измеряли.');
}
// 07
{
 const s=slide('Латентность и память','30 секунд. На одинаковых 24 звонках один запрос занимает в среднем 6,41 секунды на RTX3060 и 28,34 секунды на CPU. Загрузка модели исключена из задержки. Память учитывает пик загрузки. Пропускная способность рассчитана по среднему времени. GPU batch4 на полном val даёт 3,16 секунды на звонок в среднем, но это амортизированное время. Третью точку нельзя напрямую сравнивать с первыми двумя из-за другого набора.','reports/sprint_benchmark_cpu_pred.resources.json; reports/sprint_benchmark_cuda_pred.resources.json; reports/sprint_val_pred.resources.json');
 table(s,[['Показатель','CPU','GPU'],['Среднее на звонок','28,34 с','6,41 с'],['p95','54,39 с','8,96 с'],['Расчётно звонков/час','127','561'],['Пиковая RAM','5032 МиБ','2065 МиБ'],['Выделенная VRAM','—','2959 МиБ'],['GPU-секунды / 1000','—','≈6413*']],64,178,1152,365,[620,266,266],25);
 text(s,'Пакетная обработка GPU: 3,16 с на звонок',64,574,1120,45,29,C.blue,true);
 footer(s,'i7-12700H и RTX 3060 Laptop 6 ГБ. 24 dev: batch1, 383 val: batch4.\n* Приближение по времени выполнения на выделенной GPU.');
}
// 08
{
 const s=slide('Сильные стороны и ошибки','35 секунд. Статусы заявок извлекаются лучше остальных типов: precision93,38 и recall91,75. Основные резервы: даты тарифов и состояний услуг, точность договорённостей. Высокий Temporal договорённостей считается только на сопоставленной части, поэтому не компенсирует precision38,88. Подробные grounding-метрики по типам есть в резервных слайдах.','reports/sprint_val_metrics.json');
 table(s,[['Тип факта','Precision','Recall','Temporal','State'],['Состояние услуги','58,71%','48,34%','26,14%','33,66%'],['Тариф','75,00%','55,48%','21,31%','47,17%'],['Статус заявки','93,38%','91,75%','41,22%','84,62%'],['Действие','52,67%','42,93%','—','—'],['Договорённость','38,88%','44,98%','87,77%','—']],64,184,1152,365,[360,198,198,198,198],24);
 text(s,'Приоритет: даты услуг и тарифов, ложные договорённости',64,583,1140,45,29,C.blue,true);
 footer(s,'Прочерк означает, что оценщик не применяет метрику к этому типу фактов.');
}
// 09
{
 const s=slide('Проверки и неопределённость дат','35 секунд. 92unit-теста и 11проверок реального API прошли. На новом локальном наборе 27из43 строгих сценариев вместо16. Stress-группа пока6из15. Для прошлого месяца API показывает диапазон, а для неназванного начала unknown. В отдельных категориях конкурсное ISO-поле содержит статистическую точку внутри периода, которая явно помечена как оценка.','reports/sprint_api_check.json; reports/sprint_new_suite_gpu_report.json; src/compactfacts/date_evidence.py');
 text(s,'92',64,184,190,95,77,C.blue,true);text(s,'unit-теста пройдены',64,280,500,42,27);
 text(s,'11',64,366,190,90,77,C.blue,true);text(s,'проверок HTTP API пройдены',64,460,530,60,27);
 text(s,'«В прошлом месяце»',700,190,510,70,32,C.ink,true);
 text(s,'1 октября – 1 ноября\nпри звонке 14 ноября',700,280,500,95,30,C.blue);
 text(s,'Начало не названо',700,419,510,50,30,C.ink,true);text(s,'unknown',700,482,500,55,38,C.blue,true);
 footer(s,'Локальный сложный набор: 27/43. В stress-группе: 6/15. Скрытый adversarial-тест недоступен.');
}
// 10
{
 const s=slide('Демонстрация: история заявки','45 секунд с показом браузера. Открыть http://127.0.0.1:8001/. Использовать reports/defense/explicit_transition.json. Показать два состояния одной заявки: принята31марта до4апреля, вработе с4апреля без окончания. Источники2и8. Затем при наличии времени показать appointment.json: будущий четверг19ноября14–18. При проблеме сети демонстрация локальная, сервер не требует внешнего LLM. Не использовать старый шумный transition.json как пример точного начала: там оно не сказано.','reports/defense/explicit_transition.json; reports/sprint_api_check.json');
 text(s,'«Заявка 5160 была принята 31 марта».\n«С 4 апреля заявка 5160 в работе».',64,181,1152,116,32);
 table(s,[['Статус','Начало','Окончание','Источник'],['принята','31.03.2026','04.04.2026','реплика 2'],['в работе','04.04.2026','открыто','реплика 8']],64,348,1152,207,[360,264,264,264],25);
 text(s,'Демо: http://127.0.0.1:8001/',64,595,1152,45,29,C.blue,true);
}
// 11
{
 const s=slide('Развитие на реальных данных','40 секунд. Эти пункты являются следующим этапом, а не реализованным производственным циклом. Self-training: сохранять обезличенные реальные обращения, брать только уверенные псевдометки и проверять источник. Неоднозначные факты выборочно отдавать большой модели. Новый адаптер выпускать только после фиксированного dev. Для новых атрибутов расширять схему и целевые примеры. Русский проверен, для других языков нужны отдельные данные и оценка.','docs/DEFENSE.md; docs/SPRINT.md');
 const rows=[['Self-training','Псевдоразметка уверенных фактов с проверкой источника'],['Выборочная проверка LLM','Спорные даты, отрицания и подтверждения'],['Новые атрибуты','Расширение схемы и целевые обучающие примеры'],['Новые языки','Языковые данные и отдельная проверка качества']];
 rows.forEach((a,i)=>{const y=182+i*102;text(s,a[0],64,y,1120,42,30,C.blue,true);text(s,a[1],64,y+47,1120,43,26);});
 footer(s,'План: новый адаптер проходит неизменный dev, после чего становится доступен откат.');
}
// 12
{
 const s=slide('', '20 секунд. Мы подготовили работающий компактный прототип, который проходит оба порога на полном открытом val. Он работает локально на CPU и одной небольшой GPU. Честные ограничения: небольшой запас по датам, неполная стресс-устойчивость и отсутствие независимого скрытого результата. Готовы показать демо и ответить на вопросы.','reports/SPRINT_REPORT.md. Иллюстрация: сгенерированная командная символика.',true);coverImage(s);
 text(s,'ZooBears',64,135,700,110,88,C.white,true);
 text(s,'45,89%',68,298,630,90,70,'#79C8FF',true);text(s,'Temporal при пороге 45,06%',68,386,650,45,28,'#DBE8FC');
 text(s,'52,97%',68,463,630,90,70,'#79C8FF',true);text(s,'State при пороге 51,45%',68,552,650,45,28,'#DBE8FC');
 text(s,'Открытый val, 383 звонка',68,602,750,32,22,'#AFC2DD');text(s,'github.com/djakaranda/ZooBears',68,646,950,35,24,'#AFC2DD');all[11].title='ZooBears: результаты';
}
// 13 backup
{
 const s=slide('Резерв: все метрики по типам','Резерв. Grounding turn договорённостей44,65%: модель часто указывает иную реплику подтверждения, чем gold. Высокий span не означает полную точность извлечения. Не сравнивать локальные43случая с полным val.','reports/sprint_val_metrics.json');
 table(s,[['Атрибут','P','R','Temporal','State','Реплика','Span'],['Услуга','58,71','48,34','26,14','33,66','85,84','92,81'],['Тариф','75,00','55,48','21,31','47,17','77,78','90,91'],['Заявка','93,38','91,75','41,22','84,62','93,19','99,03'],['Действие','52,67','42,93','—','—','93,67','98,13'],['Договорённость','38,88','44,98','87,77','—','44,65','100,00']],64,193,1152,375,[260,128,128,159,159,159,159],23);
 footer(s,'Все значения в процентах. P = precision, R = recall. Прочерк: метрика неприменима.');
}
// 14
{
 const s=slide('Резерв: параметры обучения','Резерв. База заморожена. Обучаем 10 092 544 параметра LoRA, rank16 alpha32 dropout0.05 all-linear. Оптимизатор предыдущего прогона не переносим. Выбираем step100 по полному dev. Полный train содержит1 826 213токенов, целевых165730. Не обрезали обучающие ответы.','artifacts/qwen_sprint/adapter_config.json; artifacts/qwen_sprint/experiment.json; reports/sprint_training_data_audit.json');
 table(s,[['Параметр','Значение'],['Base / LoRA','596 049 920 / 10 092 544'],['LoRA','rank16, alpha32, dropout0,05, all-linear'],['Оптимизатор','AdamW, LR2e-5, weight decay0,01'],['Batch / accumulation','1 / 8'],['Точность и память','BF16 CUDA, gradient checkpointing'],['Лимиты входа / ответа','3072 / 512 токенов'],['Seed / эпохи / выбор','73 / одна полная / step100 на dev']],64,174,1152,450,[420,732],23);
}
// 15
{
 const s=slide('Резерв: определения метрик','Резерв. Precision1040из1727, recall1040из1845. Temporal290из632 сопоставленных фактов с восстанавливаемой датой, даты сравниваются по дню. State330из623. Реплика837из1040. Span612из631: только случаи с правильной репликой и спанами у обеих сторон, нужно непустое пересечение. Текущий val уже анализировали, не называть его независимым скрытым тестом.','data/raw/case_metrics.py; reports/sprint_val_metrics.json; reports/sprint_date_boundary_metrics.json');
 table(s,[['Метрика','Что проверяет оценщик'],['Precision / Recall','1040 совпадений из 1727 ответов / 1845 gold'],['Temporal','290 из 632. Начало и конец совпадают по дню'],['State','330 из 623 текущих состояний'],['Grounding turn','837 из 1040 сопоставленных фактов'],['Grounding span','Пересечение при верной реплике и двух spans']],64,182,1152,370,[380,772],24);
 text(s,'Открытый val уже участвовал в анализе ошибок',64,582,1152,45,28,C.blue,true);
 footer(s,'Независимое подтверждение даст скрытый тест жюри.');
}
// 16
{
 const s=slide('Резерв: ресурсные режимы','Резерв. Первые два режима измерены на одном dev-наборе24звонка. Третий на полном val383. Поэтому это инвентаризация режимов, а не строгая сопоставимая кривая качество–ресурсы. GPU-секунды на1000 вbatch1 примерно6413 по synchronizedwalltime, вbatch4 3162. Нет сравнения с GPU-временем учителя и нет измерения денежной экономии.','reports/sprint_benchmark_cpu_pred.resources.json; reports/sprint_benchmark_cuda_pred.resources.json; reports/sprint_val_pred.resources.json');
 table(s,[['Режим','Набор','Среднее','Звонков/ч','VRAM'],['CPU batch1','24 dev','28,34 с','127','—'],['GPU batch1','те же 24','6,41 с','561','2959 МиБ'],['GPU batch4','383 val','3,16 с*','1139','2891 МиБ']],64,184,1152,280,[320,222,190,220,200],24);
 text(s,'* Амортизированное время в пакете',64,509,1152,43,27,C.blue,true);
 text(s,'Для строгой кривой нужен одинаковый набор во всех режимах.',64,572,1152,65,25);
}
// 17
{
 const s=slide('Резерв: готовность по критериям','Резерв. Не выставлять себе баллы жюри. Формальные пороги открытого val выполнены и размер соответствует диапазону. Dockerfile есть, образ не проверен. По глубине57сравнений, ошибки по типам и ограничения сохранены, но финальная кривая не полностью сопоставима. Эта презентация отражает актуальный релиз. Self-training и мультиязычность представлены как планы, не реализованные функции.','docs/Критерии_оценивания_МТС_MWS_CompactFactExtraction.pdf; reports/SPRINT_REPORT.md');
 table(s,[['Критерий','Подтверждение и границы'],['Качество','Пороги val пройдены, параметрическая ветка обоснована'],['Техническая реализация','API, тесты, CPU/GPU. Docker-образ не проверен'],['Глубина','57 вариантов, ошибки по типам. Кривая частично сопоставима'],['Презентация и защита','Актуальные цифры, живое демо, резервные пояснения'],['Идеи развития','Self-training, спорные факты, новые атрибуты и языки']],64,182,1152,388,[374,778],23);
 footer(s,'Скрытый adversarial-тест и окончательную оценку проводит жюри.');
}
// 18
{
 const s=slide('Резерв: сокращение числа параметров','Резерв. В критерии достаточно уменьшения числа параметров минимум в10раз ИЛИ GPU-секунд минимум в5раз. Организаторы называют учителем GLM-5.3. По сохранённым официальным публичным метаданным всего753,33млрд параметров. Консервативный расчёт учитывает только down-проекции экспертов основной сети241,59млрд. Верхняя оценка студента с LoRA и CPU-моделями610млн. Отношение396,1. Архитектура учителя MoE, полный размер не равен вычислениям на токен. Серверная ревизия учителя неизвестна. Этот расчёт не доказывает ускорение или денежную экономию.','reports/TEACHER_PARAMETER_COMPARISON.md; reports/teacher_parameter_comparison.json; reports/glm53_public_metadata.json');
 table(s,[['Что сравниваем','Параметры'],['GLM-5.3, опубликованный размер','753,33 млрд'],['Консервативная часть весов учителя','241,59 млрд'],['Верхняя оценка всей нашей системы','0,61 млрд'],['Консервативное отношение','≥396 раз при требовании ≥10']],64,180,1152,330,[750,402],25);
 text(s,'Сравнение относится к размеру модели',64,550,1152,45,29,C.blue,true);
 footer(s,'Учитель использует MoE. Серверная ревизия неизвестна. Ускорение и денежную экономию не измеряли.');
}
const timing=[15,25,35,35,45,30,30,30,30,45,35,25];
let elapsed=0;
for(let i=0;i<12;i++){
 const start=elapsed;elapsed+=timing[i];
 const stamp=n=>`${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
 all[i].note=`${stamp(start)}–${stamp(elapsed)}. `+all[i].note.replace(/^\d+ секунд(?: с показом браузера)?\.\s*/, '');
 presentation.slides.items[i].speakerNotes.text=all[i].note+'\n\nИсточники: '+all[i].source;
}
presentation.slides.items[17].speakerNotes.text+='\nhttps://huggingface.co/zai-org/GLM-5.3';
await fs.writeFile(path.join(dir,'source.json'),JSON.stringify(presentation.toProto()));
await fs.writeFile(path.join(root,'output/presentation/ZooBears_speaker_notes.md'),'# ZooBears: заметки к защите\n\nОсновные слайды: 1–12, план на 6 минут 20 секунд. Запас до 7 минут: 40 секунд. Вопросы: 5 минут, резервные слайды 13–18.\n\n'+all.map((a,i)=>`## ${i+1}. ${a.title}\n\n${a.note}\n\nИсточник: ${a.source}\n`).join('\n'));
const candidate=path.join(dir,'candidate.pptx');await(await PresentationFile.exportPptx(presentation)).save(candidate);
console.log('DRAFT EXPORTED',all.length);
for(let i=0;i<presentation.slides.items.length;i++){
 const png=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(dir,'previews',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));console.log('RENDER',i+1);
}
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'output/presentation/ZooBears_defense_2026.pptx'),pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(dir,'final.v2.validation.json')});
console.log(JSON.stringify(result));

